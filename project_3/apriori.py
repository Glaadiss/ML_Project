from project_3.data import get_nominal_df
from apyori import apriori

(transactions, df) = get_nominal_df(3)
rules = apriori(transactions, min_support=0.10, min_confidence=0.7)
results = list(rules)


def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:
            conf = o.confidence
            supp = r.support
            x = ", ".join(list(o.items_base))
            y = ", ".join(list(o.items_add))
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)" % (x, y, supp, conf))
            frules.append((x, y))
    return frules


# Print rules found in the courses file.
print_apriori_rules(results)
