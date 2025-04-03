import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec

# === PARAMÈTRES GÉNÉRAUX ===
chemin_fichier = r'C:\Users\Simon\Documents\ArkeaAM\VSCode\AES\data.xlsx'
onglet = 'data'
duree_annees = 8
jours_par_an = 365
nb_jours = duree_annees * jours_par_an
nb_scenarios_bootstrap = 200
nb_scenarios_mc = 200

verbose = True

# === STRUCTURES PRODUITS ===
produits = {
    'SPCEPAB': {'indice': 'SPCEPAB Index', 'allocation': 0.25, 'coupon': 0.08, 'barriere_coupon': 0.68, 'barriere_capital': 0.60},
    'SPFRPAB': {'indice': 'SPFRPAB Index', 'allocation': 0.25, 'coupon': 0.08, 'barriere_coupon': 0.70, 'barriere_capital': 0.60},
    'SPXFP':   {'indice': 'SPXFP Index',   'allocation': 0.25, 'coupon': 0.075,'barriere_coupon': 0.80, 'barriere_capital': 0.70},
    'FRDEV40': {'indice': 'FRDEV40 Index', 'allocation': 0.15, 'coupon': 0.08, 'barriere_coupon': 0.72, 'barriere_capital': 0.60},
    'BFRTEC10':{'indice': 'BFRTEC10 Index','allocation': 0.10, 'coupon': 0.065,'barriere_coupon': 4.5,  'barriere_capital': None}
}

# === CHARGEMENT & NETTOYAGE DES DONNÉES ===
df = pd.read_excel(chemin_fichier, sheet_name=onglet, parse_dates=['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

# === FONCTION STRUCTURÉ ===
def simulate_product(start_date, end_date, index_values, coupon, bar_coupon, bar_capital, is_rate=False):
    initial = index_values.loc[start_date]
    coupons = 0
    for year in range(1, duree_annees + 1):
        obs_date = start_date + pd.DateOffset(years=year)
        if obs_date > index_values.index[-1]:
            break
        obs_date = index_values.index[index_values.index.get_indexer([obs_date], method='bfill')[0]]
        level = index_values.loc[obs_date]
        ratio = level if is_rate else level / initial
        if (is_rate and ratio <= bar_coupon) or (not is_rate and ratio >= bar_coupon):
            coupons += coupon
    final_date = index_values.index[index_values.index.get_indexer([end_date], method='bfill')[0]]
    final_level = index_values.loc[final_date]
    perf = 1
    if not is_rate and final_level / initial < bar_capital:
        perf = final_level / initial
    return coupons, coupons + perf - 1

# === BACKTEST HISTORIQUE ===
backtest_results = []
for start_date in df.index:
    end_date = start_date + pd.DateOffset(years=duree_annees)
    if end_date > df.index[-1]:
        break
    total_perf = 0
    for key, prod in produits.items():
        data = df[prod['indice']]
        coupons, perf = simulate_product(
            start_date, end_date, data,
            prod['coupon'], prod['barriere_coupon'], prod['barriere_capital'],
            is_rate=(key == 'BFRTEC10')
        )
        total_perf += prod['allocation'] * perf
    tri = (1 + total_perf) ** (1 / duree_annees) - 1
    backtest_results.append({'Date de lancement': start_date, 'TRI': tri})

df_backtest = pd.DataFrame(backtest_results)

# === STAT HELPERS ===
def summary_stats(data):
    return {
        'Moyenne': np.mean(data),
        'Médiane': np.median(data),
        'Min': np.min(data),
        'Max': np.max(data),
        'VaR 5%': np.percentile(data, 5),
        'VaR 95%': np.percentile(data, 95)
    }

# === BOOTSTRAP SCENARIOS ===
rendements = df.pct_change().dropna()
mouvements_taux = df['BFRTEC10 Index'].diff().dropna()
last_values = df.iloc[-1]
simus_bootstrap = []

if verbose:
    print(f"[BOOTSTRAP] Démarrage de {nb_scenarios_bootstrap} simulations...")

for i in range(nb_scenarios_bootstrap):
    if verbose and i % (nb_scenarios_bootstrap // 10) == 0:
        print(f"[BOOTSTRAP] Progression : {int(i / nb_scenarios_bootstrap * 100)}%...")

    scenario = pd.DataFrame()
    for key, prod in produits.items():
        indice = prod['indice']
        if key == 'BFRTEC10':
            delta = np.random.choice(mouvements_taux, size=nb_jours, replace=True)
            taux_path = [last_values[indice]]
            for d in delta:
                taux_path.append(taux_path[-1] + d)
            scenario[indice] = taux_path[1:]
        else:
            ret = np.random.choice(rendements[indice], size=nb_jours, replace=True)
            values = [last_values[indice]]
            for r in ret:
                values.append(values[-1] * (1 + r))
            scenario[indice] = values[1:]
    scenario.index = pd.date_range(start=df.index[-1], periods=nb_jours, freq='B')

    total_perf = 0
    for key, prod in produits.items():
        coupons, perf = simulate_product(
            scenario.index[0], scenario.index[-1], scenario[prod['indice']],
            prod['coupon'], prod['barriere_coupon'], prod['barriere_capital'],
            is_rate=(key == 'BFRTEC10')
        )
        total_perf += prod['allocation'] * perf
    tri = (1 + total_perf) ** (1 / duree_annees) - 1
    simus_bootstrap.append(tri)

if verbose:
    print("[BOOTSTRAP] Terminé ✅")

# === MONTE CARLO (loi log-normale & Ornstein-Uhlenbeck) ===

mu_sigma = {
    col: (rendements[col].mean(), rendements[col].std())
    for col in rendements.columns if col != 'BFRTEC10 Index'
}

def calibrate_ou_model(series):
    dt = 1
    r = series.values
    dr = np.diff(r)
    r_t = r[:-1]
    
    b, a = np.polyfit(r_t, dr, 1)
    
    theta = -b
    mu = a / theta if theta != 0 else np.mean(r)
    sigma = np.std(dr - (a + b * r_t))
    
    return theta, mu, sigma

simus_mc = []

if verbose:
    print(f"[MONTE CARLO] Calibration OU sur la série de taux...")

theta, mu_ou, sigma_ou = calibrate_ou_model(df['BFRTEC10 Index'])

if verbose:
    print(f"[MONTE CARLO] Démarrage de {nb_scenarios_mc} simulations...")

for i in range(nb_scenarios_mc):
    if verbose and i % (nb_scenarios_mc // 10) == 0:
        print(f"[MONTE CARLO] Progression : {int(i / nb_scenarios_mc * 100)}%...")

    scenario = pd.DataFrame()
    for key, prod in produits.items():
        indice = prod['indice']
        
        if key == 'BFRTEC10':
            taux_path = [last_values[indice]]
            for _ in range(nb_jours):
                r_prev = taux_path[-1]
                dr = theta * (mu_ou - r_prev) + sigma_ou * np.random.normal()
                taux_path.append(r_prev + dr)
            scenario[indice] = taux_path[1:]

        else:
            mu, sigma = mu_sigma[indice]
            ret = np.random.normal(mu, sigma, size=nb_jours)
            values = [last_values[indice]]
            for r in ret:
                values.append(values[-1] * (1 + r))
            scenario[indice] = values[1:]

    scenario.index = pd.date_range(start=df.index[-1], periods=nb_jours, freq='B')

    total_perf = 0
    for key, prod in produits.items():
        coupons, perf = simulate_product(
            scenario.index[0], scenario.index[-1], scenario[prod['indice']],
            prod['coupon'], prod['barriere_coupon'], prod['barriere_capital'],
            is_rate=(key == 'BFRTEC10')
        )
        total_perf += prod['allocation'] * perf
    tri = (1 + total_perf) ** (1 / duree_annees) - 1
    simus_mc.append(tri)

if verbose:
    print("[MONTE CARLO] Terminé ✅")

# === CALCUL STATS ===
stats_hist = summary_stats(df_backtest['TRI'])
stats_boot = summary_stats(simus_bootstrap)
stats_mc = summary_stats(simus_mc)

# === AFFICHAGE GRAPHIQUES ===
fig = plt.figure(constrained_layout=True, figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig)

def plot_table(ax, stats, title):
    ax.axis('off')
    data = [[k, f"{v*100:.2f}%"] for k, v in stats.items()]
    table = ax.table(cellText=data, colLabels=["Statistique", "Valeur"], loc='center', cellLoc='center')
    table.scale(1, 1.5)
    ax.set_title(title, fontsize=11)

# === BACKTEST ===
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df_backtest['Date de lancement'], df_backtest['TRI'], marker='o', color='tab:blue')
ax1.set_title("TRI Backtest Historique")
ax1.set_ylabel("TRI annualisé")
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.grid(True)

# === BOOTSTRAP ===
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(np.array(simus_bootstrap) * 100, bins=50, edgecolor='black', color='tab:orange')
ax2.set_title(f"Bootstrap ({nb_scenarios_bootstrap} scénarios)")
ax2.set_xlabel("TRI annualisé (%)")
ax2.set_ylabel("Fréquence")
ax2.grid(True)

# === MONTE CARLO ===
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(np.array(simus_mc) * 100, bins=50, edgecolor='black', color='tab:green')
ax3.set_title(f"Monte Carlo log-normal ({nb_scenarios_mc} scénarios)")
ax3.set_xlabel("TRI annualisé (%)")
ax3.set_ylabel("Fréquence")
ax3.grid(True)

# === TABLES DE STATS ===
plot_table(fig.add_subplot(gs[1, 0]), stats_hist, "Stats Backtest")
plot_table(fig.add_subplot(gs[1, 1]), stats_boot, "Stats Bootstrap")
plot_table(fig.add_subplot(gs[1, 2]), stats_mc, "Stats Monte-Carlo")

plt.suptitle("Analyse complète du portefeuille de structurés (3 méthodes)", fontsize=16)
plt.show()

nb_trajectoires = 5
trajectoires = []
start_value = last_values['BFRTEC10 Index']

for _ in range(nb_trajectoires):
    path = [start_value]
    for _ in range(nb_jours):
        r_prev = path[-1]
        dr = theta * (mu_ou - r_prev) + sigma_ou * np.random.normal()
        path.append(r_prev + dr)
    trajectoires.append(path)

# === PLOT DES TRAJECTOIRES DE TAUX ===
plt.figure(figsize=(10, 5))
for path in trajectoires:
    plt.plot(path)
plt.title("Simulation du taux 10 ans via Ornstein-Uhlenbeck (5 trajectoires)")
plt.xlabel("Jours (252 x 8 = 2016)")
plt.ylabel("Taux simulé")
plt.axhline(mu_ou, linestyle='--', color='gray', label=f"μ (moy. réversion) = {mu_ou:.2f}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()