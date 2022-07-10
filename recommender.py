import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## Import required libraries/packages
import os
import numpy as np
import pandas as pd
from termcolor import colored
from tabulate import tabulate
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

## Print a break for interface
bashCommand = "clear"
os.system(bashCommand)

## Binary matrix encoder
te = TransactionEncoder()

## Preview of some user numbers
user_no_examples = [
    "YGC", "D3J", "PAO", "W03", "I94"
]

df_full = pd.read_csv("random_data.csv", index_col=0)
df_full["date"] = pd.to_datetime(df_full["date"])

########################################
############ FUNCTIONS #################
########################################

## Preview all transactions of the user
def show_transaction_preview(user_no):
    carts = df_full[df_full["user"] == user_no].sort_values("date", ascending=False).head(20)
    return carts

## Check whether user is new or existing
def is_existing_user(user_no):
    existing_users = df_full["user"].drop_duplicates().to_list()
    if user_no in existing_users:
        return True
    else:
        return False

## Recommender function
def apriori_recommendation(user_no):
    last_buys = df_full[df_full["user"] == user_no]
    binary_matrix = last_buys.groupby(['transaction_id']).item.agg(list).reset_index()['item'].to_list()
    binary_matrix = te.fit(binary_matrix).transform(binary_matrix)
    binary_matrix = pd.DataFrame(binary_matrix, columns=te.columns_)
    data = apriori( binary_matrix,
                    min_support=0.001,
                    use_colnames=True).sort_values( "support",
                                                    ascending=False).reset_index(drop=True)
    data = [list(x) for x in data["itemsets"]]
    data = [x for sublist in data for x in sublist]
    data = list(dict.fromkeys(data))[:10]
    if len(last_buys) >= 10:
        for i in last_buys["item"].to_list():
            while len(data) < 10:
                data.append(i)
    return data

## Top sellers
def get_top():
    df = df_full.groupby("item").quantity.sum()
    df = pd.DataFrame(df.sort_values(ascending=False)).reset_index()

    df["quantity"] = df["quantity"].astype(int)
    df = df[df["quantity"] > 1]
    df = df["item"].to_list()

    return df

########################################
########### INTRODUCTION ###############
########################################

## Print example of user numbers
print(colored("Some example of user numbers are: ", "yellow"))
print(colored(pd.Series(user_no_examples), "red"))
print("\n")

## Get user input
user_no = str(input("Enter the user number: "))
print("\n")
print(colored(f">>> User number selected as '{user_no}'.", "green"))
print("\n")

## Show user the preview of the user transactions
print(colored(  f"User with user number '{user_no}' has transactions as shown below (only latest transactions are shown).",
                "yellow"))
print(colored(  "Table shows top 20 latest transactions.",
                "yellow"))
example_transactions = show_transaction_preview(user_no)
if len(example_transactions) != 0:
    print(colored(  tabulate(   example_transactions,
                                example_transactions.columns,
                                tablefmt="pretty"),
                    "cyan"))
else:
    print("\n")
    print(colored(  f"!!! > Transaction history information is not available for user with user number '{user_no}'.",
                    "red"))
print("\n")

## Get recommendations
if is_existing_user(user_no):
    recommendation = apriori_recommendation(user_no)

else:
    print(colored(  f"Since transaction history information is not available for user with user number '{user_no}', top sellers will be recommended.",
                    "red"))
    recommendation = get_top()

recommendation_table = pd.DataFrame({"recommended_items": recommendation})

## Show recommended items
print(colored(f"For the user '{user_no}', the recommendation will show these items:", "yellow"))
print(colored(tabulate(recommendation_table, recommendation_table.columns, tablefmt="pretty"), "cyan"))