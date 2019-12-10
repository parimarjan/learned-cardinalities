
### analyzing costs
# TODO: do this based on result logs
# all_opt_plans = defaultdict(list)
# all_est_plans = defaultdict(list)
# num_opt_plans = []
# num_est_plans = []
# for i, _ in enumerate(opt_costs):
    # opt_cost = opt_costs[i]
    # est_cost = est_costs[i]
    # # plot both optimal, and estimated plans
    # explain = est_plans[i]
    # leading = get_leading_hint(explain)
    # opt_explain = opt_plans[i]
    # opt_leading = get_leading_hint(opt_explain)
    # sql = queries[i]["sql"]
    # template = queries[i]["template_name"]

    # all_est_plans[leading].append((template, deterministic_hash(sql), sql))
    # all_opt_plans[opt_leading].append((template, deterministic_hash(sql), sql))

# print("num opt plans: {}, num est plans: {}".format(\
        # len(all_opt_plans), len(all_est_plans)))

## saving per bucket summaries
# print(all_opt_plans.keys())
# for k,v in all_opt_plans.items():
    # num_opt_plans.append(len(v))
# for k,v in all_est_plans.items():
    # num_est_plans.append(len(v))

# print(sorted(num_opt_plans, reverse=True)[0:10])
# print(sorted(num_est_plans, reverse=True)[0:10])

# with open("opt_plan_summaries.pkl", 'wb') as fp:
    # pickle.dump(all_opt_plans, fp, protocol=pickle.HIGHEST_PROTOCOL)

# with open("est_plan_summaries.pkl", 'wb') as fp:
    # pickle.dump(all_est_plans, fp, protocol=pickle.HIGHEST_PROTOCOL)

# pdb.set_trace()
