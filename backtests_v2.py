import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# === PARAMÈTRES GÉNÉRAUX ===
chemin_fichier = r'C:\Users\Simon\Documents\ArkeaAM\VSCode\AES\data.xlsx'
onglet = 'data'
duree_annees = 8
jours_par_an = 365
verbose = True

# === OPTIONS ===
apply_fees = True
management_fee_rate = 0.015

# === STRUCTURE DES PRODUITS ===
produits = {
    'SPCEPAB': {'indice': 'SPCEPAB Index', 'allocation': 0.25, 'coupon': 0.08, 'barriere_coupon': 0.68},
    'SPFRPAB': {'indice': 'SPFRPAB Index', 'allocation': 0.25, 'coupon': 0.08, 'barriere_coupon': 0.70},
    'SPXFP':   {'indice': 'SPXFP Index',   'allocation': 0.25, 'coupon': 0.075, 'barriere_coupon': 0.80},
    'FRDEV40': {'indice': 'FRDEV40 Index', 'allocation': 0.15, 'coupon': 0.08, 'barriere_coupon': 0.72},
    'BFRTEC10':{'indice': 'BFRTEC10 Index','allocation': 0.10, 'coupon': 0.065, 'barriere_coupon': 4.5}
}

# === CHARGEMENT ET NETTOYAGE DES DONNÉES ===
df = pd.read_excel(chemin_fichier, sheet_name=onglet, parse_dates=['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

# === BACKTEST HISTORIQUE TRI PAR SIMULATION JOURNALIÈRE ===
backtest_results = []
daily_fee = management_fee_rate / jours_par_an

for start_date in df.index:
    end_date = start_date + pd.DateOffset(years=duree_annees)
    if end_date > df.index[-1]:
        break

    # Vérification si tous les coupons sont payés
    total_coupon = 0
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

    valeur_portefeuille = 100.0
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    for current_date in date_range:
        for year in range(1, duree_annees + 1):
            coupon_date = start_date + pd.DateOffset(years=year)
            if current_date.date() == coupon_date.date():
                for key, prod in produits.items():
                    valeur_portefeuille += prod['coupon'] * prod['allocation'] * 100

        if apply_fees:
            valeur_portefeuille *= (1 - daily_fee)

    final_value = valeur_portefeuille
    tri = (final_value / 100) ** (1 / duree_annees) - 1
    backtest_results.append({'Date de lancement': start_date, 'TRI': tri})

# === DATAFRAME DES TRI ===
df_backtest = pd.DataFrame(backtest_results)

# === SIMULATION ÉVOLUTION VL POUR UN SCÉNARIO ===
start_date = df_backtest['Date de lancement'].iloc[-1]
end_date = start_date + pd.DateOffset(years=duree_annees)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

valeur_portefeuille = 100.0
vl_series = []

for current_date in date_range:
    for year in range(1, duree_annees + 1):
        coupon_date = start_date + pd.DateOffset(years=year)
        if current_date.date() == coupon_date.date():
            for key, prod in produits.items():
                valeur_portefeuille += prod['coupon'] * prod['allocation'] * 100

    if apply_fees:
        valeur_portefeuille *= (1 - daily_fee)

    vl_series.append((current_date, valeur_portefeuille))

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

# === CALCUL TRI DE LA COURBE DE DROITE ===
final_value = vl_series[-1][1]
tri_simule = (final_value / 100) ** (1 / duree_annees) - 1

# === AFFICHAGE GRAPHIQUES ===
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(16, 7))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])

# Graph 1 : TRI historique
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df_backtest['Date de lancement'], df_backtest['TRI'], marker='o', color='darkblue')
ax1.set_title("TRI annualisé - Backtest historique")
ax1.set_xlabel("Date de lancement")
ax1.set_ylabel("TRI")
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.set_ylim(0, 0.08)
ax1.grid(True)

# Stats sous le graphique
ax_stats = fig.add_subplot(gs[1, 0])
ax_stats.axis('off')
table_data = [[k, f"{v*100:.2f}%"] for k, v in stats_hist.items()]
ax_stats.table(cellText=table_data, colLabels=["Statistique", "Valeur"], loc='center', cellLoc='center')

# Graph 2 : évolution de la VL
ax2 = fig.add_subplot(gs[:, 1])
dates = [x[0] for x in vl_series]
values = [x[1] for x in vl_series]
ax2.plot(dates, values, label='Valeur portefeuille', color='darkgreen')
ax2.axhline(100, linestyle='--', color='gray', label='Base 100')
ax2.set_title(f"Évolution VL (lancement {start_date.date()})")
ax2.set_xlabel("Date")
ax2.set_ylabel("Valeur")
ax2.grid(True)
ax2.legend()

# Afficher la valeur finale et le TRI associé
ax2.text(dates[-1], values[-1], f"{values[-1]:.2f} (TRI: {tri_simule*100:.2f}%)", 
         verticalalignment='center', fontsize=10, color='black')

plt.tight_layout()
plt.show()