import os
import subprocess
from pddl import parse_domain, parse_problem

FAST_DOWNWARD_ALIAS = "seq-opt-fdss-1"
TIME_LIMIT = 10


class pddlsim(object):
    def __init__(self, domain_file):
        self.domain_file = domain_file

    def action_to_list(self, action):
        return action.replace("(", "").replace(")", "").split(" ")

    def plan(self, problem_file):
        try:
            output = subprocess.check_output(
                f"python ./downward/fast-downward.py --alias {FAST_DOWNWARD_ALIAS} "
                + f"--search-time-limit {TIME_LIMIT} "
                + "--plan-file pddl_output.txt "
                # + f"--sas-file aaa.sas "
                + f"{self.domain_file} {problem_file}",
                shell=True,
            )
        except subprocess.CalledProcessError:
            print ("something went wrong")
            return None
        output = output.decode("utf-8").split("\n")
        start_index = next(
            (i + 1 for i, s in enumerate(output) if "Actual search time" in s), None
        )
        end_index = next((i for i, s in enumerate(output) if "Plan length" in s), None)
        plan = output[start_index:end_index]
        for action_id, action in enumerate(plan):
            plan[action_id] = action.split(" ")
        return plan

    def get_init_states(self, problem_file):  # assuming there is no comments
        with open(problem_file) as f:
            problem = f.readlines()
        init_start = next((i + 1 for i, s in enumerate(problem) if "init" in s), None)
        init_end = next((i - 2 for i, s in enumerate(problem) if "goal" in s), None)

        init_states = [fact.strip() for fact in problem[init_start:init_end]]
        return init_states

    def get_intermediate_states(self, problem_file, plan_file):
        """apply a plan to a pddl problem using the validation tool, return a list of intermediate states"""
        int_states = []
        try:
            output = subprocess.check_output(
                "VAL/build/linux64/Release/bin/Validate -v "
                + f"{self.domain_file} {problem_file} {plan_file}",
                shell=True,
            )
        except:
            return None
        init_states = self.get_init_states(problem_file)
        output = output.decode("utf-8").split("\n")
        start_index = next((i + 2 for i, s in enumerate(output) if "-----" in s), None)
        end_index = next(
            (i for i, s in enumerate(output) if "Plan executed successfully" in s),
            None,
        )
        state_changes = output[start_index:end_index]
        state_i = 0
        while state_i < len(state_changes):
            state = state_changes[state_i]
            if "Checking" in state:
                int_states.append(init_states.copy())
            elif "Adding" in state:
                formatted = " ".join(state.split(" ")[1:])
                if formatted not in init_states:
                    init_states.append(formatted)
            elif "Deleting" in state:
                formatted = " ".join(state.split(" ")[1:])
                if formatted in init_states:
                    init_states.remove(formatted)
            state_i += 1
        int_states.append(init_states.copy())
        # int_states.pop(0)
        return int_states

    def get_preconditions_by_action(self, action):
        action_name = action[0]
        actions = [x for x in parse_domain(self.domain_file).actions]
        preconditions = [ac.precondition for ac in actions if ac.name == action_name][
            0
        ].operands
        param_vars = [ac.parameters for ac in actions if ac.name == action_name][0]
        return self.reformat_fact_using_values(action, param_vars, preconditions)

    def get_effects_by_states(self, prev_states, cur_states):
        effects = []
        # if facts got added:
        for fact in cur_states:
            if fact not in prev_states:
                formatted_fact = fact.replace("(", "")
                formatted_fact = formatted_fact.replace(")", "")
                formatted_fact = formatted_fact.split(" ")
                effects.append(formatted_fact)
        # if facts got removed:
        for fact in prev_states:
            if fact not in cur_states:
                formatted_fact = fact.replace("(", "")
                formatted_fact = formatted_fact.replace(")", "")
                formatted_fact = formatted_fact.split(" ")
                formatted_fact.insert(0, "not")
                effects.append(formatted_fact)
        return effects
        # action_name = action[0]
        # actions = [x for x in parse_domain(self.domain_file).actions]
        # effects = [ac.effect for ac in actions if ac.name == action_name][
        #     0
        # ].operands
        # param_vars = [ac.parameters for ac in actions if ac.name == action_name][0]
        # return self.reformat_fact_using_values(action, param_vars, effects)

        # TODO: use state change to determine effects

    def reformat_fact_using_values(self, action, param_vars, facts):
        action_params = action[1:-1]
        reformat_pres = []

        for fact in facts:
            reformat_pre = []
            if not hasattr(fact, "name"):
                fact = fact.argument
                reformat_pre.append("not")
            predicate = fact.name
            reformat_pre.append(predicate)
            parameters = fact.terms
            for param in parameters:
                reformat_pre.append(action_params[param_vars.index(param)])
            reformat_pres.append(reformat_pre)
        return reformat_pres


if __name__ == "__main__":
    task = "domains/boil_water_in_the_microwave/"
    task = "domains/cook_a_frozen_pie/"
    task = "domains/halve_an_egg/"
    # task = "domains/bringing_water/"
    # task = "domains/store_beer/"
    # task = "domains/store_brownies/"
    task = "domains/store_firewood/"
    p = task + "problem.pddl"
    # p = "updated_problem.pddl"
    test = pddlsim(task + "domain.pddl")
    test.plan(p)
    # print (test.get_intermediate_states(p, "pddl_output.txt"))
