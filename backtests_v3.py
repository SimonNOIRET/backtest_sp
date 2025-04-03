import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec

# === PARAMÈTRES GÉNÉRAUX ===
chemin_fichier = r'C:\Users\Simon\Documents\ArkeaAM\VSCode\AES\data.xlsx'
onglet = 'data'
duree_annees = 7
jours_par_an = 365
verbose = True

# === OPTIONS ===
apply_fees = True
use_estr = True
management_fee_rate = 0.015

# === STRUCTURE DES PRODUITS ===
produits = {
    'SPCEPAB': {'indice': 'SPCEPAB Index', 'allocation': 0.25, 'coupon': 0.08, 'barriere_coupon': 0.68},
    'SPFRPAB': {'indice': 'SPFRPAB Index', 'allocation': 0.25, 'coupon': 0.08, 'barriere_coupon': 0.70},
    'SPXFP':   {'indice': 'SPXFP Index',   'allocation': 0.25, 'coupon': 0.075, 'barriere_coupon': 0.80},
    'FRDEV40': {'indice': 'FRDEV40 Index', 'allocation': 0.15, 'coupon': 0.08, 'barriere_coupon': 0.72},
    'BFRTEC10':{'indice': 'BFRTEC10 Index','allocation': 0.10, 'coupon': 0.065, 'barriere_coupon': 4.5}
}

# === CHARGEMENT DES DONNÉES ===
df = pd.read_excel(chemin_fichier, sheet_name=onglet, parse_dates=['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
df.columns = df.columns.str.strip()
df['ESTR'] = df['ESTRON Index'] / 100

# === FONCTION DE SIMULATION VL ===
def simulate_vl_with_estr(df, produits, start_date, duree_annees=8):
    end_date = start_date + pd.DateOffset(years=duree_annees)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Dates ajustées de versement des coupons
    coupon_dates = []
    for year in range(1, duree_annees + 1):
        target = start_date + pd.DateOffset(years=year)
        if target > df.index[-1]:
            continue
        coupon_date = df.index[df.index.get_indexer([target], method='bfill')[0]]
        coupon_dates.append(coupon_date)

    VL = 100.0
    cash_cap = 0.0
    results = []

    for i, current_date in enumerate(date_range):
        if current_date not in df.index:
            continue

        total_coupon = 0
        if current_date in coupon_dates:
            for key, prod in produits.items():
                total_coupon += prod['coupon'] * prod['allocation'] * 100

        frais = -management_fee_rate * VL / jours_par_an if apply_fees else 0.0
        estr_rate = df.loc[current_date - pd.Timedelta(days=1), 'ESTR'] if i > 0 and current_date - pd.Timedelta(days=1) in df.index else 0.0
        cash_inter = cash_cap + total_coupon + frais
        cash_cap = cash_inter * (1 + estr_rate / 360)
        VL = 100 + cash_cap

        results.append({
            'Date': current_date,
            'VL': VL,
            'Cash_cap': cash_cap,
            'Cash_inter': cash_inter,
            'Frais': frais,
            'Coupon': total_coupon,
            'ESTR (veille)': estr_rate
        })

    return pd.DataFrame(results)

# === BACKTEST HISTORIQUE ===
backtest_results = []

for start_date in df.index:
    end_date = start_date + pd.DateOffset(years=duree_annees)
    if end_date > df.index[-1]:
        continue

    valid = True
    for key, prod in produits.items():
        idx_series = df[prod['indice']]
        start_val = idx_series.loc[start_date]
        for year in range(1, duree_annees + 1):
            obs_date = start_date + pd.DateOffset(years=year)
            if obs_date > idx_series.index[-1]:
                valid = False
                break
            obs_date = idx_series.index[idx_series.index.get_indexer([obs_date], method='bfill')[0]]
            obs_val = idx_series.loc[obs_date]
            ratio = obs_val if key == 'BFRTEC10' else obs_val / start_val
            if (key == 'BFRTEC10' and ratio > prod['barriere_coupon']) or (key != 'BFRTEC10' and ratio < prod['barriere_coupon']):
                valid = False
                break
        if not valid:
            break

    if not valid:
        continue

    df_vl = simulate_vl_with_estr(df, produits, start_date, duree_annees)
    final_value = df_vl['VL'].iloc[-1]
    tri = (final_value / 100) ** (1 / duree_annees) - 1
    backtest_results.append({'Date de lancement': start_date, 'TRI': tri})

# === DATAFRAME DES TRI ===
df_backtest = pd.DataFrame(backtest_results)

# === SIMULATION ÉVOLUTION VL POUR UN SCÉNARIO ===
start_date = df_backtest['Date de lancement'].iloc[-1]
df_simulation = simulate_vl_with_estr(df, produits, start_date, duree_annees)

# === STATS TRI ===
def summary_stats(data):
    return {
        'Moyenne': np.mean(data),
        'Médiane': np.median(data),
        'Min': np.min(data),
        'Max': np.max(data),
        'VaR 5%': np.percentile(data, 5),
        'VaR 95%': np.percentile(data, 95)
    }

stats_hist = summary_stats(df_backtest['TRI'])
final_value = df_simulation['VL'].iloc[-1]
tri_simule = (final_value / 100) ** (1 / duree_annees) - 1

# === AFFICHAGE GRAPHIQUES ===
fig = plt.figure(figsize=(16, 7))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])

# Graph 1 : TRI historique
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df_backtest['Date de lancement'], df_backtest['TRI'], marker='o', color='darkblue')
ax1.set_title("TRI annualisé - Arkéa Engagement Structurés - Backtest historique")
ax1.set_xlabel("Date de lancement")
ax1.set_ylabel("TRI")
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.set_ylim(0, df_backtest['TRI'].max() + 0.04)
ax1.grid(True)

# Stats sous le graphique
ax_stats = fig.add_subplot(gs[1, 0])
ax_stats.axis('off')
table_data = [[k, f"{v*100:.2f}%"] for k, v in stats_hist.items()]
ax_stats.table(cellText=table_data, colLabels=["Statistique", "Valeur"], loc='center', cellLoc='center')

# Graph 2 : évolution de la VL
ax2 = fig.add_subplot(gs[:, 1])
ax2.plot(df_simulation['Date'], df_simulation['VL'], label='Simulation du comportement de la VL hors MtM', color='darkgreen')
ax2.axhline(100, linestyle='--', color='gray', label='Base 100')
ax2.set_title(f"\u00c9volution VL (lancement {start_date.date()})")
ax2.set_xlabel("Date")
ax2.set_ylabel("Valeur")
ax2.grid(True)
ax2.legend()

# Annotations des coupons
for i, row in df_simulation.iterrows():
    if row['Coupon'] > 0:
        ax2.annotate(f"+{row['Coupon']:.2f}", xy=(row['Date'], row['VL']),
                     xytext=(0, 8), textcoords='offset points', fontsize=8,
                     ha='center', color='blue')

# Afficher la valeur finale et le TRI associé
ax2.text(df_simulation['Date'].iloc[-1], df_simulation['VL'].iloc[-1],
         f"{final_value:.2f} (TRI: {tri_simule*100:.2f}%)",
         verticalalignment='center', fontsize=10, color='black')

plt.tight_layout()
plt.show()

