import pandas as pd
from datetime import datetime
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('OnlineRetail.csv', delimiter=';')
df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)

df['UnitPrice'] = df['UnitPrice'].str.replace(',', '.').astype(float)
df = df.dropna()
df=df[df['Quantity'] > 0]
df=df[df['UnitPrice'] > 0]

df_aus = (df[df['Country'] =="Australia"]
 .groupby(['InvoiceNo', 'Description'])['Quantity']
 .sum().unstack().reset_index()
 .set_index('InvoiceNo'))

df_ger = (df[df['Country'] =="Germany"]
 .groupby(['InvoiceNo', 'Description'])['Quantity']
 .sum().unstack().reset_index()
 .set_index('InvoiceNo'))

df_jap = (df[df['Country'] =="Japan"]
 .groupby(['InvoiceNo', 'Description'])['Quantity']
 .sum().unstack().reset_index()
 .set_index('InvoiceNo'))



records = []
for i in range(0, 57):
    items = []
    for j in range(0, 500):
        item = str(df_aus.values[i,j])
        if(item != 'nan'):
            items.append(item)
    records.append(items)

te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
store_data_transformed = pd.DataFrame(te_ary, columns=te.columns_)

from apyori import apriori
association_rules = apriori(records, min_support=0.15, min_confidence=0.4, min_lift=4, min_length=2)
association_results = list(association_rules)

#Lihat jumlah aturan asosiasi yg dihasilkan
print(len(association_results))

#Lihat aturan asosiasi pertama (ke-0)
print(association_results[0])

#Print semua aturan asosiasi beserta metrik-nya:
for item in association_results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")


######### 
# Analisis data untuk mendapatkan aturan asosiasi dengan memanfaatkan 
# algoritma FP-Growth. 

# Buat frequent items sets dgn min support tertentu
frequent_itemsets_fpgrowth = fpgrowth(store_data_transformed, min_support=0.1, use_colnames=True)

from mlxtend.frequent_patterns import association_rules

#Buat aturan-aturan asosiasi, disaring berdasar nilai confidence tertentu
rules_conf_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence",min_threshold=0.52, support_only = False)

#Buat aturan-aturan asosiasi, disaring berdasar nilai lift tertentu
rules_lift_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="lift",min_threshold=4, support_only = False)

#Hasil analisis aturan asosiasi dengan fpgrowth (tersimpan sbg dataframe)
rules_conf = rules_conf_fpgrowth[['antecedents', 'consequents']]
rules_lift = rules_lift_fpgrowth[['antecedents', 'consequents']]


###############
# Eksperimen Perbandingan Waktu eksekusi algoritma Apriori vs FP-Growth

#Applying Apriori
apriori_start = datetime.now()
apriori_frequent_itemsets = apriori(store_data_transformed, min_support=0.1, use_colnames=True)
apriori_end = datetime.now()
running_time = apriori_end-apriori_start
print("Apriori Running Time: ", str(running_time))

#Applying FP-Growth
fpgrowth_start = datetime.now()
fpgworth_frequent_itemsets = fpgrowth(store_data_transformed, min_support=0.1, use_colnames=True)
fpgrowth_end = datetime.now()
running_time = fpgrowth_end-fpgrowth_start
print("FP-Growth Running Time: ", str(running_time))