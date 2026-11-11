"""Part 3: run evaluate_team.py on every team's file (ours included), save one result per team and the table.
python run_evaluations.py   ->  evaluation_results/<file>.json, <file>.png, scores_table.csv"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from evaluate_team import evaluate, save_result

ROOT = Path(__file__).resolve().parent
P = ROOT.parent / 'peer_projects'
TEAMS = {   # team -> its reconstruction file; add the files you get in class
    'ours (OAM)': ROOT / 'team_OAM_reconstruction.npz',
    'a15cardoso': P / 'a15cardoso_python_ws/team_03_reconstruction.npz',
    'Anandita06': P / 'Anandita06_reverse_engineering_neural_spike_simulation/team_3_reconstruction.npz',
    'Chlo3Cook': P / 'Chlo3Cook_spike-simulator-MAC/team_05_reconstruction.npz',
    'danielkwork0': P / 'danielkwork0_reverse-engineering-challenge/team_7_reconstruction.npz',
    'benji-qu': P / 'benji-qu_SpikesReconstruction/data/output/team_10_reconstruction.npz',
    'Abdulwaliy': P / 'Abdulwaliy/team_Abdulwaliy_et_Eleni_reconstruction.npz',
}

if __name__ == '__main__':
    out, rows = ROOT / 'evaluation_results', {}
    for team, f in TEAMS.items():
        if not f.exists():
            print('missing', f); continue
        r = evaluate(ROOT / 'spike_data.npz', f)
        save_result(r, out)
        m = r['metrics']
        rows[team] = {'file': f.name, 'final score': r['final_score'], 'valid': r['validity'],
                      **{k: round(v, 3) for k, v in r['scores'].items()},
                      'spike F1': round(m.get('spike F1', 0), 3), 'R2': round(m.get('R2', 0), 4),
                      't matches': r['validity_checks'].get('t equals the reference time vector')}
    table = pd.DataFrame(rows).T.sort_values('final score', ascending=False)
    table.to_csv(out / 'scores_table.csv')
    pd.set_option('display.width', 250)
    print(table.to_string())
