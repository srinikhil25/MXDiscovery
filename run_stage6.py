"""
run_stage6.py - DFT Validation with Quantum ESPRESSO

ALGORITHM:
    Stage 6 validates the top ML-screened candidates using first-principles
    DFT calculations. This provides GROUND TRUTH electronic structure data
    that our semi-empirical estimates from Stage 5 could only approximate.

    TWO MODES:
        1. "generate" - Create QE input files + master WSL run script
        2. "parse"    - After DFT completes, parse outputs and compare with ML

    WHY DFT VALIDATION?
        - ML screening (CHGNet) gives stability but NOT electronic structure
        - Semi-empirical TE estimates used literature bandgap values (not computed)
        - DFT gives the ACTUAL bandgap, DOS, band structure for each candidate
        - This is the standard in computational materials discovery papers

    WHAT WE VALIDATE:
        - Formation energy: DFT vs CHGNet (expect ~30 meV/atom agreement)
        - Bandgap: DFT vs our literature estimates (could differ significantly)
        - DOS shape: metallic vs semiconducting character
        - Band structure: effective mass, band degeneracy

    KEY INSIGHT FOR THIS PROJECT:
        We only need to run DFT on UNIQUE MXene structures, not per composite
        partner. The composite partner affects the interface properties, but
        the bare MXene electronic structure is what we compute with DFT.
        So Mo2CO2 is computed once, and the results apply to Mo2CO2/Ag_NW,
        Mo2CO2/SWCNT, Mo2CO2/PEDOT:PSS, etc.

    EXECUTION:
        1. Generate: python run_stage6.py generate
        2. Run DFT in WSL2: cd /mnt/d/MXDiscovery/data/results/dft && bash run_all_dft.sh
        3. Parse results: python run_stage6.py parse

PSEUDOPOTENTIAL DOWNLOAD:
    Before running DFT, you need SSSP pseudopotentials.
    This script can download them automatically:
        python run_stage6.py download_pp
"""

import json
import sys
import subprocess
from pathlib import Path

from loguru import logger


def get_unique_structures(rankings: list[dict]) -> dict:
    """
    Extract unique MXene structures from ranked candidates.

    WHY:
        Multiple candidates may share the same base MXene (e.g., Mo2CO2).
        DFT is expensive, so we only compute each unique structure once.
        The composite partner affects interface/transport but not the
        bare MXene's electronic structure.

    RETURNS:
        dict mapping structure_key -> {info about the candidate}
        e.g., "Mo2C_O" -> {"mxene_formula": "Mo2C", "termination": "O", ...}
    """
    unique = {}
    for cand in rankings:
        key = f"{cand['mxene_formula']}_{cand['termination']}"
        if key not in unique:
            unique[key] = {
                "mxene_formula": cand["mxene_formula"],
                "termination": cand["termination"],
                "name": key,
                # Track which composite partners use this structure
                "partners": [],
                "ranks": [],
            }
        unique[key]["partners"].append(cand["composite_partner"])
        unique[key]["ranks"].append(cand["rank"])
    return unique


def find_cif_for_structure(relaxed_dir: Path, mxene_formula: str, termination: str) -> Path:
    """
    Find the relaxed CIF file for a given MXene structure.

    Since CIF files are named like Mo2C_O_PEDOT_PSS_relaxed.cif, we need
    to find ANY CIF with matching formula and termination (they all have
    the same MXene structure regardless of composite partner).
    """
    pattern = f"{mxene_formula}_{termination}_*_relaxed.cif"
    matches = list(relaxed_dir.glob(pattern))
    if matches:
        return matches[0]
    return None


def mode_generate(base_dir: Path, top_n: int = 5):
    """
    Generate all QE input files for the top candidates.

    WHAT IT CREATES:
        data/results/dft/
            Mo2C_O/
                Mo2C_O.scf.in       - SCF ground state
                Mo2C_O.nscf.in      - NSCF dense k-grid
                Mo2C_O.dos.in       - DOS calculation
                Mo2C_O.bands.in     - Band structure
                Mo2C_O.bands_pp.in  - Bands post-processing
            Ti2C_O/
                ...
            run_all_dft.sh          - Master WSL2 run script
    """
    from ase.io import read as ase_read
    from src.simulation.qe_manager import QEManager

    # Load rankings
    rankings_path = base_dir / "data" / "final_rankings.json"
    with open(rankings_path) as f:
        rankings = json.load(f)

    # Get top N candidates
    top_rankings = rankings[:top_n]

    # Extract unique structures
    unique = get_unique_structures(top_rankings)

    print("\n" + "=" * 70)
    print("  STAGE 6: DFT VALIDATION - INPUT GENERATION")
    print("=" * 70)
    print(f"  Top {top_n} candidates map to {len(unique)} unique MXene structures:")
    for key, info in unique.items():
        partners_str = ", ".join(info["partners"][:3])
        if len(info["partners"]) > 3:
            partners_str += f" (+{len(info['partners'])-3} more)"
        print(f"    {key}: ranks {info['ranks']} -> {partners_str}")

    # Setup QE manager
    # Use WSL-compatible paths for pseudo_dir in the input files
    wsl_pseudo = "/mnt/d/MXDiscovery/data/pseudopotentials"
    qe = QEManager(
        pseudo_dir=wsl_pseudo,
        output_dir=str(base_dir / "data" / "results" / "dft"),
        ecutwfc=60.0,
        ecutrho=480.0,
        kpoints=(8, 8, 1),
    )

    relaxed_dir = base_dir / "data" / "structures" / "relaxed"
    generated_names = []

    print("\n  Generating QE input files...")

    for key, info in unique.items():
        name = info["name"]
        formula = info["mxene_formula"]
        term = info["termination"]

        # Find the relaxed CIF
        cif_path = find_cif_for_structure(relaxed_dir, formula, term)
        if cif_path is None:
            print(f"    WARNING: No CIF found for {name}, skipping")
            continue

        # Load structure
        atoms = ase_read(str(cif_path))
        print(f"\n    {name}:")
        print(f"      Structure: {len(atoms)} atoms, cell: {atoms.get_cell().lengths()}")

        # Generate all input files
        scf_file = qe.generate_scf_input(atoms, name, calculation="scf")
        print(f"      SCF:  {scf_file.name}")

        nscf_file = qe.generate_nscf_input(atoms, name, kpoints=(16, 16, 1))
        print(f"      NSCF: {nscf_file.name}")

        dos_file = qe.generate_dos_input(name)
        print(f"      DOS:  {dos_file.name}")

        bands_file = qe.generate_bands_input(atoms, name)
        print(f"      Bands: {bands_file.name}")

        bands_pp = qe.generate_bands_pp_input(name)
        print(f"      Bands PP: {bands_pp.name}")

        generated_names.append(name)

    # Generate master run script
    if generated_names:
        master = qe.generate_master_script(generated_names, nprocs=4)
        print(f"\n  Master script: {master}")

    # Print instructions
    print("\n" + "=" * 70)
    print("  NEXT STEPS:")
    print("=" * 70)
    print("  1. Download pseudopotentials (if not already done):")
    print("       python run_stage6.py download_pp")
    print("")
    print("  2. Run DFT calculations in WSL2:")
    print("       wsl -d Ubuntu")
    print("       cd /mnt/d/MXDiscovery/data/results/dft")
    print("       bash run_all_dft.sh 2>&1 | tee dft_log.txt")
    print("")
    print("  3. After DFT completes, parse results:")
    print("       python run_stage6.py parse")
    print("")
    print(f"  Estimated time: ~{len(generated_names) * 30}-{len(generated_names) * 120} min")
    print(f"  (depends on structure size and convergence)")
    print("=" * 70)


def mode_parse(base_dir: Path):
    """
    Parse completed DFT outputs and compare with ML predictions.

    WHAT IT DOES:
        1. Parse SCF outputs for DFT total energy and Fermi energy
        2. Parse DOS outputs for bandgap detection
        3. Compare DFT formation energy vs CHGNet formation energy
        4. Compare DFT bandgap vs our literature estimate
        5. Generate validation report

    WHY THIS MATTERS:
        If DFT confirms our ML predictions (within ~30 meV/atom for energy,
        ~0.1 eV for bandgap), it validates the entire screening pipeline.
        If DFT disagrees, we learn which assumptions were wrong.
    """
    from src.simulation.qe_manager import QEManager

    qe = QEManager(
        output_dir=str(base_dir / "data" / "results" / "dft"),
    )

    # Load ML predictions for comparison
    rankings_path = base_dir / "data" / "final_rankings.json"
    screening_path = base_dir / "data" / "screening_results.json"

    with open(rankings_path) as f:
        rankings = json.load(f)
    with open(screening_path) as f:
        screening = json.load(f)

    # Build lookup for CHGNet formation energies
    chgnet_fe = {}
    for sr in screening:
        key = f"{sr['mxene_formula']}_{sr['termination']}"
        try:
            chgnet_fe[key] = float(sr["formation_energy"])
        except (ValueError, TypeError):
            pass

    # Get unique structures
    unique = get_unique_structures(rankings[:5])
    dft_dir = base_dir / "data" / "results" / "dft"

    print("\n" + "=" * 70)
    print("  STAGE 6: DFT VALIDATION - RESULTS PARSING")
    print("=" * 70)

    all_results = {}

    for key, info in unique.items():
        name = info["name"]
        calc_dir = dft_dir / name

        print(f"\n  --- {name} ---")

        # Parse SCF output
        scf_out = calc_dir / f"{name}.scf.out"
        if scf_out.exists():
            scf_results = qe.parse_scf_output(scf_out)
            print(f"    SCF: {'CONVERGED' if scf_results['converged'] else 'FAILED'}")
            if scf_results["total_energy_ev"] is not None:
                print(f"    Total energy: {scf_results['total_energy_ev']:.4f} eV")
            if scf_results["fermi_energy_ev"] is not None:
                print(f"    Fermi energy: {scf_results['fermi_energy_ev']:.4f} eV")
            if scf_results["wall_time"]:
                print(f"    Wall time: {scf_results['wall_time']}")
        else:
            scf_results = {"error": "SCF not yet run"}
            print(f"    SCF: NOT FOUND (run DFT first)")

        # Parse DOS output
        dos_file = calc_dir / f"{name}.dos"
        if dos_file.exists():
            dos_results = qe.parse_dos_output(dos_file)
            if "error" not in dos_results:
                print(f"    DOS bandgap: {dos_results['bandgap_ev']:.3f} eV")
                print(f"    Metallic: {dos_results['is_metallic']}")
                print(f"    DOS at Fermi: {dos_results['dos_at_fermi']:.4f}")
            else:
                dos_results = {}
                print(f"    DOS: {dos_results.get('error', 'parse error')}")
        else:
            dos_results = {}
            print(f"    DOS: NOT FOUND")

        # Compare with ML predictions
        ml_fe = chgnet_fe.get(key)
        if ml_fe is not None and scf_results.get("total_energy_ev") is not None:
            # Note: DFT total energy != formation energy directly
            # We'd need elemental reference energies for proper comparison
            # For now, we just report both
            print(f"\n    -- Comparison --")
            print(f"    CHGNet E_f:  {ml_fe:.4f} eV/atom")
            # DFT formation energy would need elemental references

        # ML bandgap estimate
        from src.screening.te_predictor import TEPredictor
        ml_bg = TEPredictor.BANDGAP_ESTIMATES.get(
            (info["mxene_formula"], info["termination"]), 0.3
        )
        dft_bg = dos_results.get("bandgap_ev")
        if dft_bg is not None:
            diff = abs(dft_bg - ml_bg)
            print(f"    ML est. bandgap: {ml_bg:.3f} eV")
            print(f"    DFT bandgap:     {dft_bg:.3f} eV")
            print(f"    Difference:      {diff:.3f} eV ({'GOOD' if diff < 0.15 else 'SIGNIFICANT'})")

        all_results[name] = {
            "scf": scf_results,
            "dos": dos_results,
            "chgnet_fe": ml_fe,
            "ml_bandgap": ml_bg,
            "dft_bandgap": dft_bg,
        }

    # Save parsed results
    results_path = base_dir / "data" / "dft_validation_results.json"
    # Convert to JSON-safe
    safe_results = {}
    for k, v in all_results.items():
        safe_results[k] = {}
        for k2, v2 in v.items():
            if isinstance(v2, dict):
                safe_results[k][k2] = {
                    kk: (str(vv) if not isinstance(vv, (int, float, bool, type(None), str)) else vv)
                    for kk, vv in v2.items()
                }
            else:
                safe_results[k][k2] = v2

    with open(results_path, "w") as f:
        json.dump(safe_results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)

    completed = sum(1 for r in all_results.values()
                    if r["scf"].get("converged", False))
    total = len(all_results)
    print(f"  DFT calculations completed: {completed}/{total}")

    if completed > 0:
        print("\n  If bandgap differences are < 0.15 eV: ML estimates are validated")
        print("  If bandgap differences are > 0.3 eV: re-rank candidates with DFT values")
    else:
        print("\n  No DFT results found yet. Run the calculations first:")
        print("    wsl -d Ubuntu")
        print("    cd /mnt/d/MXDiscovery/data/results/dft")
        print("    bash run_all_dft.sh")

    print("=" * 70)


def mode_download_pp(base_dir: Path):
    """
    Download SSSP pseudopotentials needed for our elements.

    SSSP (Standard Solid-State Pseudopotentials):
        - Curated library from Materials Cloud (EPFL)
        - Tested for accuracy and efficiency across the periodic table
        - PBE functional (standard for DFT calculations)
        - PAW (Projector Augmented Wave) type - high accuracy

    We only download the pseudopotentials for elements in our candidates:
        Mo, C, O, N, Ti (covers all our top MXene structures)

    SOURCE:
        https://pseudopotentials.quantum-espresso.org/
        Direct download from the QE pseudopotential repository
    """
    pp_dir = base_dir / "data" / "pseudopotentials"
    pp_dir.mkdir(parents=True, exist_ok=True)

    # Elements we need for our top candidates
    needed_elements = {
        "Mo": "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF",
        "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
        "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
        "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF",
        "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
    }

    # QE pseudopotential download base URL
    base_url = "https://pseudopotentials.quantum-espresso.org/upf_files"

    print("\n" + "=" * 70)
    print("  DOWNLOADING SSSP PSEUDOPOTENTIALS")
    print("=" * 70)

    for element, pp_file in needed_elements.items():
        pp_path = pp_dir / pp_file
        if pp_path.exists():
            print(f"  {element}: {pp_file} [ALREADY EXISTS]")
            continue

        url = f"{base_url}/{pp_file}"
        print(f"  {element}: downloading {pp_file}...")

        try:
            # Use WSL wget for reliable download
            wsl_pp_dir = f"/mnt/d/MXDiscovery/data/pseudopotentials"
            result = subprocess.run(
                ["wsl", "-d", "Ubuntu", "--", "bash", "-c",
                 f"wget -q -O '{wsl_pp_dir}/{pp_file}' '{url}' 2>&1 && echo OK || echo FAILED"],
                capture_output=True, text=True, timeout=120
            )
            output = result.stdout.strip()
            if "OK" in output and pp_path.exists():
                size_kb = pp_path.stat().st_size / 1024
                print(f"         -> OK ({size_kb:.0f} KB)")
            else:
                print(f"         -> FAILED: {output}")
                # Try alternative URL pattern
                alt_url = f"https://www.quantum-espresso.org/upf_files/{pp_file}"
                print(f"         Trying alternative URL...")
                result2 = subprocess.run(
                    ["wsl", "-d", "Ubuntu", "--", "bash", "-c",
                     f"wget -q -O '{wsl_pp_dir}/{pp_file}' '{alt_url}' 2>&1 && echo OK || echo FAILED"],
                    capture_output=True, text=True, timeout=120
                )
                if "OK" in result2.stdout and pp_path.exists():
                    size_kb = pp_path.stat().st_size / 1024
                    print(f"         -> OK ({size_kb:.0f} KB)")
                else:
                    print(f"         -> FAILED. Download manually from:")
                    print(f"            {url}")
        except Exception as e:
            print(f"         -> ERROR: {e}")
            print(f"         Download manually: {url}")

    # Verify all present
    print("\n  Verification:")
    all_ok = True
    for element, pp_file in needed_elements.items():
        pp_path = pp_dir / pp_file
        if pp_path.exists():
            print(f"    [OK] {pp_file}")
        else:
            print(f"    [MISSING] {pp_file}")
            all_ok = False

    if all_ok:
        print("\n  All pseudopotentials ready! Proceed with DFT.")
    else:
        print("\n  Some pseudopotentials missing. Download them manually.")
        print(f"  Directory: {pp_dir}")
    print("=" * 70)


def mode_run_dft(base_dir: Path):
    """
    Run DFT calculations directly via WSL2.

    This invokes the master script through WSL2 subprocess.
    Output streams to console in real time.
    """
    dft_dir = base_dir / "data" / "results" / "dft"
    master_script = dft_dir / "run_all_dft.sh"

    if not master_script.exists():
        print("ERROR: Master script not found. Run 'generate' mode first.")
        return

    wsl_script = "/mnt/d/MXDiscovery/data/results/dft/run_all_dft.sh"

    print("\n" + "=" * 70)
    print("  RUNNING DFT CALCULATIONS VIA WSL2")
    print("=" * 70)
    print("  This may take several hours. Output streams below.")
    print("  Press Ctrl+C to abort (calculations can be resumed).")
    print("=" * 70 + "\n")

    try:
        process = subprocess.Popen(
            ["wsl", "-d", "Ubuntu", "--", "bash", wsl_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
        print(f"\n  DFT calculations finished with exit code {process.returncode}")
    except KeyboardInterrupt:
        print("\n\n  Aborted by user. Partial results may be available.")
        print("  Re-run will skip completed SCF calculations (check .out files).")
    except Exception as e:
        print(f"\n  Error running WSL: {e}")
        print("  Try running manually in WSL2 terminal:")
        print(f"    cd /mnt/d/MXDiscovery/data/results/dft")
        print(f"    bash run_all_dft.sh")


if __name__ == "__main__":
    base_dir = Path("D:/MXDiscovery")

    if len(sys.argv) < 2:
        mode = "generate"
    else:
        mode = sys.argv[1].lower()

    if mode == "generate":
        mode_generate(base_dir, top_n=5)
    elif mode == "parse":
        mode_parse(base_dir)
    elif mode == "download_pp":
        mode_download_pp(base_dir)
    elif mode == "run":
        mode_run_dft(base_dir)
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python run_stage6.py [generate|parse|download_pp|run]")
        sys.exit(1)
