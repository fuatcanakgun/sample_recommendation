def sample():    
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    import pandas as pd

    te = TransactionEncoder()

    df_full = pd.read_csv("random_data.csv")
    df_agg = df_full.groupby('transaction_id').item.agg(list).reset_index(name="items")

    def get_apriori(item_no, sup):
        data = apriori(get_binary(item_no), use_colnames=True, min_support=sup).sort_values("support", ascending=False)
        return data

    def binary_sum(matrix):
        matrix = matrix.T
        matrix["sum"] = matrix.sum(axis=1)
        matrix = matrix.sort_values("sum", ascending=False)
        matrix["position"] = list(range(len(matrix)))
        return matrix

    def get_binary(item_no):
        sample_cart = [item_no]

        included_carts = []
        for basket in df_agg["items"]:
            if set(sample_cart).issubset(set(basket)):
                included_carts.append(basket)

        binary_matrix = te.fit(included_carts).transform(included_carts)
        binary_matrix = pd.DataFrame(binary_matrix, columns=te.columns_)
        return binary_matrix

    bi_mtx = get_binary("T8WS1V")

    binary_sum(bi_mtx)

    get_apriori("T8WS1V", 0.08)

def main():
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    sample()
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


main()