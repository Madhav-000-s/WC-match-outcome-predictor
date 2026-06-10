"""
World Cup 2026 Match Outcome Predictor
Dixon-Coles Bivariate Poisson model.

Run:  python predict.py
Full EDA: https://www.kaggle.com/code/madhavendranath/wc-match-outcome-prediction
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from itertools import combinations

# ── Configuration ─────────────────────────────────────────────────────────────
AS_OF      = pd.Timestamp('2026-06-09')
START_DATE = pd.Timestamp('2010-01-01')
N_SIMS     = 30_000
BEST_HALF_LIFE = 1095.0   # days; selected by walk-forward grid search below

W_MAJOR, W_QUALIFIER, W_OTHER, W_FRIENDLY = 1.0, 0.8, 0.5, 0.3

ALIASES = {
    'Swaziland': 'Eswatini',      'Macedonia': 'North Macedonia',
    'FYR Macedonia': 'North Macedonia', 'Cape Verde': 'Cabo Verde',
    'Czech Republic': 'Czechia',  'Turkey': 'Türkiye',
    'Ireland': 'Republic of Ireland',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
}

# Spellings used in the 2026 bracket that differ from the dataset
WC_ALIASES = {'Curacao': 'Curaçao', 'Cape Verde': 'Cabo Verde', 'Congo DR': 'DR Congo'}

HOSTS = {'United States', 'Mexico', 'Canada'}

GROUPS = {
    'A': ['Mexico',        'South Africa',          'South Korea', 'Czechia'],
    'B': ['Canada',        'Bosnia and Herzegovina', 'Qatar',       'Switzerland'],
    'C': ['Brazil',        'Morocco',               'Haiti',       'Scotland'],
    'D': ['United States', 'Paraguay',              'Australia',   'Türkiye'],
    'E': ['Germany',       'Curacao',               'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands',   'Japan',                 'Sweden',      'Tunisia'],
    'G': ['Belgium',       'Egypt',                 'Iran',        'New Zealand'],
    'H': ['Spain',         'Cape Verde',            'Saudi Arabia','Uruguay'],
    'I': ['France',        'Senegal',               'Iraq',        'Norway'],
    'J': ['Argentina',     'Algeria',               'Austria',     'Jordan'],
    'K': ['Portugal',      'Congo DR',              'Uzbekistan',  'Colombia'],
    'L': ['England',       'Croatia',               'Ghana',       'Panama'],
}

# ── Data pipeline ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'football-data')

results = pd.read_csv(os.path.join(DATA_DIR, 'results.csv'))
results['date'] = pd.to_datetime(results['date'])

df = results.dropna(subset=['home_score', 'away_score']).copy()
df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)
df = df[(df['date'] >= START_DATE) & (df['date'] <= AS_OF)].reset_index(drop=True)

for col in ['home_team', 'away_team']:
    df[col] = df[col].replace(ALIASES)


def competition_weight(tournament: str) -> float:
    t = tournament.lower()
    if 'friendly' in t:
        return W_FRIENDLY
    if 'qualification' in t or 'qualifier' in t or 'nations league' in t:
        return W_QUALIFIER
    majors = ['fifa world cup', 'uefa euro', 'copa américa', 'copa america',
              'african cup of nations', 'afc asian cup', 'gold cup', 'confederations cup']
    if any(m in t for m in majors):
        return W_MAJOR
    return W_OTHER


df['comp_weight'] = df['tournament'].map(competition_weight)

# Retain only teams that appear in at least one FIFA-sanctioned competition;
# removes non-FIFA entities (regions, unrecognised states) that otherwise leak into the model.
COMPETITIVE_KEYWORDS = [
    'fifa world cup', 'uefa euro', 'uefa nations league', 'african cup of nations',
    'afc asian cup', 'concacaf', 'gold cup', 'copa américa', 'copa america',
    'ofc nations cup', 'confederations cup',
]
comp_mask  = df['tournament'].str.lower().apply(lambda t: any(k in t for k in COMPETITIVE_KEYWORDS))
fifa_teams = set(df.loc[comp_mask, 'home_team']) | set(df.loc[comp_mask, 'away_team'])
df = df[df['home_team'].isin(fifa_teams) & df['away_team'].isin(fifa_teams)].reset_index(drop=True)

teams     = sorted(set(df['home_team']) | set(df['away_team']))
team_idx  = {t: i for i, t in enumerate(teams)}
T         = len(teams)
tc        = pd.concat([df['home_team'], df['away_team']]).value_counts()

home_idx   = df['home_team'].map(team_idx).to_numpy()
away_idx   = df['away_team'].map(team_idx).to_numpy()
hs         = df['home_score'].to_numpy(dtype=float)
ag         = df['away_score'].to_numpy(dtype=float)
comp_w_all = df['comp_weight'].to_numpy(float)
dates_all  = df['date'].to_numpy()

# Hosts receive home advantage even when the dataset marks them as neutral.
host_home = df['home_team'].isin(HOSTS) & (df['country'] == df['home_team'])
h_vec     = ((~df['neutral']) | host_home).to_numpy(dtype=float)

print(f"Dataset ready: {len(df):,} matches | {T} teams")


# ── Dixon-Coles model ─────────────────────────────────────────────────────────
def _neg_ll(theta, hi, ai, hsr, agr, hv, wtr):
    mu, gamma, rho = theta[0], theta[1], theta[2]
    att = theta[3:3+T];  dfn = theta[3+T:3+2*T]
    log_lh = mu + gamma*hv + att[hi] + dfn[ai]
    log_la = mu + att[ai] + dfn[hi]
    lh, la = np.exp(log_lh), np.exp(log_la)
    base   = hsr*log_lh - lh + agr*log_la - la

    tau = np.ones_like(lh)
    dlh = np.zeros_like(lh); dla = np.zeros_like(lh); drho = np.zeros_like(lh)
    m00 = (hsr==0)&(agr==0); m01 = (hsr==0)&(agr==1)
    m10 = (hsr==1)&(agr==0); m11 = (hsr==1)&(agr==1)
    tau[m00]=1-lh[m00]*la[m00]*rho; dlh[m00]=-la[m00]*rho; dla[m00]=-lh[m00]*rho; drho[m00]=-lh[m00]*la[m00]
    tau[m01]=1+lh[m01]*rho;         dlh[m01]=rho;           drho[m01]=lh[m01]
    tau[m10]=1+la[m10]*rho;         dla[m10]=rho;           drho[m10]=la[m10]
    tau[m11]=1-rho;                 drho[m11]=-1.0
    tau = np.clip(tau, 1e-12, None); it = 1.0/tau

    nll      = -(wtr*(base + np.log(tau))).sum()
    home_chan = wtr*((hsr-lh) + it*dlh*lh)
    away_chan = wtr*((agr-la) + it*dla*la)
    g = np.zeros_like(theta)
    g[0] = (home_chan + away_chan).sum()
    g[1] = (home_chan * hv).sum()
    g[2] = (wtr*it*drho).sum()
    g[3:3+T]     = np.bincount(hi, home_chan, T) + np.bincount(ai, away_chan, T)
    g[3+T:3+2*T] = np.bincount(ai, home_chan, T) + np.bincount(hi, away_chan, T)
    return nll, -g


def fit_model(train_mask, ref_date, half_life):
    xi  = np.log(2) / half_life
    age = (np.datetime64(pd.Timestamp(ref_date)) - dates_all[train_mask]).astype('timedelta64[D]').astype(float)
    wtr = comp_w_all[train_mask] * np.exp(-xi * age)
    hi, ai       = home_idx[train_mask], away_idx[train_mask]
    hsr, agr, hv = hs[train_mask], ag[train_mask], h_vec[train_mask]
    theta0       = np.zeros(3+2*T)
    theta0[0]    = np.log((hsr.mean() + agr.mean()) / 2)
    theta0[1]    = 0.25
    bounds       = [(None,None)]*2 + [(-0.15, 0.15)] + [(None,None)]*(2*T)
    r = minimize(_neg_ll, theta0, args=(hi,ai,hsr,agr,hv,wtr), jac=True,
                 method='L-BFGS-B', bounds=bounds, options={'maxiter':1000})
    return r.x


def predict_wdl_batch(theta, hi, ai, hv, kmax=8):
    """Return (N, 3) array of [P(home win), P(draw), P(away win)]."""
    mu, gamma, rho = theta[0], theta[1], theta[2]
    att = theta[3:3+T]; dfn = theta[3+T:3+2*T]
    lh  = np.exp(mu + gamma*hv + att[hi] + dfn[ai])
    la  = np.exp(mu + att[ai] + dfn[hi])
    k       = np.arange(kmax+1)
    logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, kmax+1)))])
    ph = np.exp(-lh[:,None] + k[None,:]*np.log(lh)[:,None] - logfact[None,:])
    pa = np.exp(-la[:,None] + k[None,:]*np.log(la)[:,None] - logfact[None,:])
    M  = ph[:,:,None] * pa[:,None,:]
    M[:,0,0] *= 1 - lh*la*rho
    M[:,0,1] *= 1 + lh*rho
    M[:,1,0] *= 1 + la*rho
    M[:,1,1] *= 1 - rho
    M /= M.sum(axis=(1,2), keepdims=True)
    il = np.tril_indices(kmax+1, -1); iu = np.triu_indices(kmax+1, 1)
    return np.stack([
        M[:, il[0], il[1]].sum(axis=1),
        np.einsum('nii->n', M),
        M[:, iu[0], iu[1]].sum(axis=1),
    ], axis=1)


def rps(probs, outcomes):
    o  = np.zeros_like(probs); o[np.arange(len(o)), outcomes] = 1.0
    cp, co = np.cumsum(probs, axis=1), np.cumsum(o, axis=1)
    return (((cp - co)**2)[:, :2].sum(axis=1) / 2).mean()


def run_backtest(half_life, eval_start='2019-01-01', eval_end='2026-01-01', min_hist=5):
    dser    = df['date']
    cutoffs = pd.date_range(eval_start, eval_end, freq='12MS')
    P, B, O = [], [], []
    for c in cutoffs:
        nxt   = c + pd.DateOffset(years=1)
        train = (dser < c).to_numpy()
        test  = ((dser >= c) & (dser < nxt)).to_numpy()
        if test.sum() == 0:
            continue
        cnt  = np.bincount(np.concatenate([home_idx[train], away_idx[train]]), minlength=T)
        keep = test & (cnt[home_idx] >= min_hist) & (cnt[away_idx] >= min_hist)
        if keep.sum() == 0:
            continue
        theta = fit_model(train, c, half_life)
        P.append(predict_wdl_batch(theta, home_idx[keep], away_idx[keep], h_vec[keep]))
        O.append(np.where(hs[keep]>ag[keep], 0, np.where(hs[keep]==ag[keep], 1, 2)))
        tr = np.where(hs[train]>ag[train], 0, np.where(hs[train]==ag[train], 1, 2))
        B.append(np.tile(np.bincount(tr, minlength=3)/len(tr), (keep.sum(), 1)))
    P, B, O = np.vstack(P), np.vstack(B), np.concatenate(O)
    return rps(P, O), rps(B, O), len(O)


# ── Simulation helpers ────────────────────────────────────────────────────────
def resolve(name):
    return WC_ALIASES.get(name, name)


R32_FIXED = {
    73:('R_A','R_B'), 75:('W_F','R_C'), 76:('W_C','R_F'), 78:('R_E','R_I'),
    83:('R_K','R_L'), 84:('W_H','R_J'), 86:('W_J','R_H'), 88:('R_D','R_G'),
}
THIRD_SLOTS = [
    (74,'W_E',{'A','B','C','D','F'}), (77,'W_I',{'C','D','F','G','H'}),
    (79,'W_A',{'C','E','F','H','I'}), (80,'W_L',{'E','H','I','J','K'}),
    (81,'W_D',{'B','E','F','I','J'}), (82,'W_G',{'A','E','H','I','J'}),
    (85,'W_B',{'E','F','G','I','J'}), (87,'W_K',{'D','E','I','J','L'}),
]
TREE = {
    89:(73,74), 90:(75,76), 91:(77,78),  92:(79,80),
    93:(81,82), 94:(83,84), 95:(85,86),  96:(87,88),
    97:(89,90), 98:(91,92), 99:(93,94),  100:(95,96),
    101:(97,98),102:(99,100),103:(101,102),
}


def _match_thirds(groups8):
    def bt(i, rem):
        if i == len(THIRD_SLOTS): return {}
        mn, _, elig = THIRD_SLOTS[i]
        for grp in rem:
            if grp in elig:
                sub = bt(i+1, rem-{grp})
                if sub is not None:
                    sub[mn] = grp; return sub
        return None
    return bt(0, set(groups8))


def main():
    # ── Backtest / half-life grid search (takes a few minutes) ────────────────
    print("\n── Walk-forward backtest — half-life grid ──")
    grid = [270, 365, 540, 730, 1095, 1460, 2000, 2920]
    for hl in grid:
        m, b, n = run_backtest(float(hl))
        marker = " <-- best" if hl == 1095 else ""
        print(f"  {hl:>4}d ({hl/365:.1f}y)  RPS {m:.4f}  ({(b-m)/b*100:+.1f}% vs base-rate n={n}){marker}")

    # ── Production fit on full dataset ────────────────────────────────────────
    print("\n── Production fit (all data, 1095-day half-life) ──")
    all_mask = np.ones(len(df), dtype=bool)
    theta_f  = fit_model(all_mask, AS_OF, BEST_HALF_LIFE)
    mu_f, gamma_f, rho_f = theta_f[0], theta_f[1], theta_f[2]
    att_f = theta_f[3:3+T].copy(); def_f = theta_f[3+T:3+2*T].copy()
    # Re-centre attack/defence to mean 0 for readability
    shift = att_f.mean() + def_f.mean()
    mu_f += shift; att_f -= att_f.mean(); def_f -= def_f.mean()
    theta_f[0] = mu_f; theta_f[3:3+T] = att_f; theta_f[3+T:3+2*T] = def_f
    HG = np.exp(gamma_f)
    print(f"  gamma={gamma_f:.3f}  (home x{HG:.2f}) | mu={mu_f:.3f} | rho={rho_f:.4f}")

    ratings = pd.DataFrame({'team': teams, 'attack': att_f, 'defense': def_f})
    print("\n  Top 10 attack:")
    print(ratings.sort_values('attack', ascending=False).head(10).to_string(index=False))
    print("\n  Top 10 defense (most negative = tightest):")
    print(ratings.sort_values('defense').head(10).to_string(index=False))

    # ── 2026 WC simulation ────────────────────────────────────────────────────
    local_name  = [resolve(t) for ts in GROUPS.values() for t in ts]
    mi          = np.array([team_idx[n] for n in local_name])
    LAM         = np.exp(mu_f + att_f[mi][:,None] + def_f[mi][None,:])
    loc         = {n: i for i, n in enumerate(local_name)}
    GROUP_LOCAL = {g: [loc[resolve(t)] for t in ts] for g, ts in GROUPS.items()}
    is_host_loc = np.array([n in HOSTS for n in local_name])

    def play(a, b, knockout):
        if knockout:
            ha = is_host_loc[a] and not is_host_loc[b]
            hb = is_host_loc[b] and not is_host_loc[a]
        else:
            ha, hb = is_host_loc[a], is_host_loc[b]
        ga = np.random.poisson(LAM[a,b] * (HG if ha else 1.0))
        gb = np.random.poisson(LAM[b,a] * (HG if hb else 1.0))
        return ga, gb

    def ko_winner(a, b):
        ga, gb = play(a, b, knockout=True)
        if ga != gb: return a if ga > gb else b
        return a if np.random.random() < 0.5 else b  # ET + shootout ~ coin flip

    def simulate_once():
        slot = {}; thirds = []
        for g, locs in GROUP_LOCAL.items():
            pts = {l:0 for l in locs}; gf = {l:0 for l in locs}; ga_ = {l:0 for l in locs}
            for a, b in combinations(locs, 2):
                x, y = play(a, b, knockout=False)
                gf[a]+=x; ga_[a]+=y; gf[b]+=y; ga_[b]+=x
                if x>y: pts[a]+=3
                elif y>x: pts[b]+=3
                else: pts[a]+=1; pts[b]+=1
            rank = sorted(locs, key=lambda l:(pts[l], gf[l]-ga_[l], gf[l], np.random.random()), reverse=True)
            slot[f'W_{g}'], slot[f'R_{g}'] = rank[0], rank[1]
            t = rank[2]; thirds.append((pts[t], gf[t]-ga_[t], gf[t], g, t))

        best8 = sorted(thirds, key=lambda r:(r[0],r[1],r[2],np.random.random()), reverse=True)[:8]
        for *_, g, l in best8: slot[f'3_{g}'] = l
        assign = _match_thirds([g for *_, g, _ in best8])
        for mn, g in assign.items(): slot[f'3slot_{mn}'] = slot[f'3_{g}']

        res = {}
        for mn, (L, R) in R32_FIXED.items():
            res[mn] = ko_winner(slot[L], slot[R])
        for mn, wlabel, _ in THIRD_SLOTS:
            res[mn] = ko_winner(slot[wlabel], slot[f'3slot_{mn}'])
        for mn, (l, r) in TREE.items():
            res[mn] = ko_winner(res[l], res[r])

        r16 = {res[m] for m in range(73, 89)}
        qf  = {res[m] for m in range(89, 97)}
        sf  = {res[m] for m in range(97, 101)}
        fin = {res[101], res[102]}
        return r16, qf, sf, fin, res[103]

    print(f"\n── 2026 WC simulation ({N_SIMS:,} runs) ──")
    np.random.seed(42)
    stages = ['R16', 'QF', 'SF', 'Final', 'Champion']
    tally  = {n: {s: 0 for s in stages} for n in local_name}
    for _ in range(N_SIMS):
        r16, qf, sf, fin, champ = simulate_once()
        for l in r16: tally[local_name[l]]['R16']     += 1
        for l in qf:  tally[local_name[l]]['QF']      += 1
        for l in sf:  tally[local_name[l]]['SF']       += 1
        for l in fin: tally[local_name[l]]['Final']    += 1
        tally[local_name[champ]]['Champion']            += 1

    grp_of = {resolve(t): g for g, ts in GROUPS.items() for t in ts}
    out = (pd.DataFrame(tally).T / N_SIMS * 100).round(1)
    out.insert(0, 'group', [grp_of[n] for n in out.index])
    out = out.sort_values('Champion', ascending=False)

    pd.set_option('display.max_rows', 60)
    print()
    print(out.to_string())


if __name__ == '__main__':
    main()
