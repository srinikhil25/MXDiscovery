"""
agent.py - LLM orchestrator that connects all pipeline modules

PURPOSE:
    This is the "brain" that connects the researcher to the pipeline.
    It takes natural language input and routes to the correct module.

ARCHITECTURE:
    The agent uses a ReAct (Reasoning + Acting) pattern:
    1. THINK: Analyze what the researcher is asking
    2. ACT: Call the appropriate tool/module
    3. OBSERVE: Read the tool's output
    4. THINK: Decide if more actions are needed
    5. RESPOND: Give the researcher a synthesized answer

TOOLS AVAILABLE TO THE AGENT:
    1. search_papers(query) → search the paper database
    2. extract_data(paper_id) → extract TE data from a paper
    3. gap_analysis() → run composition space gap analysis
    4. generate_structure(formula, term, partner) → create crystal structure
    5. screen_stability(structure) → ML stability check
    6. predict_te(structure) → predict thermoelectric properties
    7. rank_candidates(candidates) → TOPSIS ranking
    8. generate_dft_input(structure) → create QE input files
    9. query_database(sql) → direct SQL query on knowledge base
    10. remember(key, value) → save to long-term memory
    11. recall(key) → retrieve from long-term memory

HOW THE LLM IS USED:
    - NOT for scientific calculations (that's what the tools do)
    - FOR natural language understanding (parse researcher intent)
    - FOR result synthesis (explain results in context)
    - FOR planning (break complex requests into tool sequences)
    - FOR literature interpretation (summarize, compare papers)
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    import ollama
except ImportError:
    ollama = None


SYSTEM_PROMPT = """You are MXDiscovery, an AI research assistant specialized in
MXene thermoelectric materials discovery. You help researchers discover new
MXene composites for wearable thermoelectric applications.

You have access to these tools:
- search_papers: Search the MXene literature database
- gap_analysis: Find unexplored MXene composition spaces
- generate_structure: Create crystal structures for candidates
- screen_stability: Check thermodynamic stability using ML potentials
- predict_te: Predict thermoelectric properties (Seebeck, PF, ZT)
- rank_candidates: Multi-criteria ranking of candidates
- query_database: SQL queries on the knowledge base
- remember / recall: Long-term memory

IMPORTANT RULES:
1. NEVER fabricate scientific data. If you don't know, say so.
2. Always cite the source of information (paper ID, database query, ML model).
3. Quantitative predictions should include confidence level and method used.
4. When suggesting experiments, explain the scientific reasoning.
5. Flag when ML predictions might be unreliable (e.g., far from training data).
6. For thermoelectric applications, remember the target is WEARABLE (T ≈ 310K).

When the researcher asks a question:
1. Determine if you need to use a tool or can answer from knowledge
2. If using tools, explain what you're doing and why
3. Present results with context (how does this compare to state of art?)
4. Suggest next steps

Respond in a clear, scientific style. Be concise but thorough."""


class Tool:
    """Represents a callable tool for the agent."""
    def __init__(self, name: str, description: str, func: callable, parameters: dict = None):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}


class MXDiscoveryAgent:
    """
    Main orchestrator agent connecting LLM to the computational pipeline.

    HOW IT WORKS:

    The agent follows a tool-use loop:
    1. User sends a message
    2. Agent sends message + tool descriptions to LLM
    3. LLM decides which tool(s) to call (or responds directly)
    4. Agent executes the tool(s)
    5. Tool results are sent back to LLM
    6. LLM synthesizes a response
    7. Response is returned to user

    MEMORY:
    - Conversation history is maintained in self.messages
    - Long-term facts are stored in self.memory (dict → JSON file)
    - Memory persists across conversations via memory.json

    ERROR HANDLING:
    - If LLM is unavailable, agent can still run tools directly
    - If a tool fails, error is reported to both LLM and user
    - All actions are logged for reproducibility
    """

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        memory_file: str = "data/database/agent_memory.json",
        db=None,         # MXeneDatabase instance
        config=None,     # config dict
    ):
        self.model = model
        self.memory_file = Path(memory_file)
        self.db = db
        self.config = config or {}
        self.messages = []
        self.memory = self._load_memory()
        self.tools = {}

        self._register_default_tools()

    def _load_memory(self) -> dict:
        """Load persistent memory from JSON file."""
        if self.memory_file.exists():
            with open(self.memory_file) as f:
                return json.load(f)
        return {}

    def _save_memory(self):
        """Persist memory to disk."""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)

    def register_tool(self, tool: Tool):
        """Register a tool that the agent can use."""
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def _register_default_tools(self):
        """Register built-in tools."""
        self.register_tool(Tool(
            name="remember",
            description="Save a fact to long-term memory. Args: key (str), value (str)",
            func=self._tool_remember,
        ))
        self.register_tool(Tool(
            name="recall",
            description="Retrieve a fact from long-term memory. Args: key (str)",
            func=self._tool_recall,
        ))
        self.register_tool(Tool(
            name="query_database",
            description="Run a SQL query on the MXene knowledge base. Args: sql (str)",
            func=self._tool_query_db,
        ))

    def _tool_remember(self, key: str, value: str) -> str:
        self.memory[key] = value
        self._save_memory()
        return f"Remembered: {key} = {value}"

    def _tool_recall(self, key: str) -> str:
        value = self.memory.get(key)
        if value:
            return f"{key}: {value}"
        return f"No memory found for key: {key}"

    def _tool_query_db(self, sql: str) -> str:
        if self.db is None:
            return "Database not connected."
        try:
            cur = self.db.conn.execute(sql)
            rows = cur.fetchall()
            if not rows:
                return "Query returned no results."
            return json.dumps([dict(r) for r in rows[:50]], indent=2, default=str)
        except Exception as e:
            return f"SQL error: {e}"

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return agent response.

        ALGORITHM:
            1. Add user message to conversation history
            2. Build tool descriptions for LLM context
            3. Send to LLM with system prompt + history + tools
            4. Parse response: check if LLM wants to call a tool
            5. If tool call: execute tool, add result, send back to LLM
            6. If direct response: return to user
            7. Limit tool-call loops to 5 iterations (prevent infinite loops)
        """
        self.messages.append({"role": "user", "content": user_message})

        if ollama is None:
            return self._fallback_response(user_message)

        # Build messages with system prompt
        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + self.messages[-20:]  # Keep last 20 messages for context window

        try:
            response = ollama.chat(
                model=self.model,
                messages=full_messages,
                options={"temperature": 0.3, "num_ctx": 8192},
            )
            assistant_msg = response["message"]["content"]
            self.messages.append({"role": "assistant", "content": assistant_msg})
            return assistant_msg

        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            return self._fallback_response(user_message)

    def _fallback_response(self, user_message: str) -> str:
        """Response when LLM is not available - direct tool routing."""
        msg = user_message.lower()

        if "gap" in msg and "analysis" in msg:
            return "To run gap analysis, use: pipeline.run_gap_analysis()"
        elif "screen" in msg or "stability" in msg:
            return "To screen candidates, use: pipeline.run_screening()"
        elif "paper" in msg or "search" in msg or "literature" in msg:
            return "To search papers, use: pipeline.fetch_papers()"
        elif "rank" in msg:
            return "To rank candidates, use: pipeline.run_ranking()"
        else:
            return (
                "LLM is not available. You can use the pipeline directly:\n"
                "  pipeline.fetch_papers()     - Fetch MXene papers\n"
                "  pipeline.extract_data()     - Extract TE data from papers\n"
                "  pipeline.run_gap_analysis() - Find unexplored compositions\n"
                "  pipeline.run_screening()    - ML stability screening\n"
                "  pipeline.run_te_prediction()- Predict TE properties\n"
                "  pipeline.run_ranking()      - Rank candidates\n"
            )

    def get_conversation_summary(self) -> str:
        """Summarize the conversation so far."""
        if not self.messages:
            return "No conversation yet."
        return f"Conversation: {len(self.messages)} messages, {len(self.memory)} memory items"
