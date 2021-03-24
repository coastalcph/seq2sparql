import json
from collections import defaultdict

templates_list = []
question_mod_entities_list = []
template_to_sparql = defaultdict(list)
template_to_question = defaultdict(list)
complexity_to_template = defaultdict(list)
template_to_complexity = defaultdict(list)
sparql_to_template = defaultdict(list)
complexity_to_mod_entities = defaultdict(list)
complexity_to_brackets = defaultdict(list)

"""
Example of "complexity" information in data
{
"complexityMeasures": {
	"npAndNestingDepth": 0,
	"parseTreeLeafCount": 9,
	"parseTreeRuleCount": 28,
	"questionTemplate": "Did [entity] 's [ADJECTIVE_SIMPLE] [ROLE_SIMPLE] [VP_SIMPLE] and [VP_SIMPLE] [entity]",
	"recursionDepth": 28,
	"sparqlMaximumChainLength": 2,
	"sparqlMaximumDegree": 4,
	"sparqlNumConstraints": 4,
	"sparqlNumVariables": 1
	},
"""

def load_data():
    num_skipped = 0
    with open("dataset.json", "r") as injson:
        inlist = json.load(injson)
        total_entries = len(inlist)
        print(f"* Len of dataset: {total_entries}")
        for i, lil_d in enumerate(inlist):
            if i % 10000:
                print(f"{i}/{total_entries}")
            try:
                q_template = lil_d["complexityMeasures"]["questionTemplate"]
                q_complexity = lil_d["complexityMeasures"]["recursionDepth"]
                sparql_pattern = lil_d["sparqlPattern"]
                question = lil_d["question"]
                questionModEntities = lil_d["questionPatternModEntities"]
                brackets = lil_d["questionWithBrackets"]
                #keep the templates
                templates_list.append(q_template)
                #put to dict
                template_to_sparql[q_template].append(sparql_pattern)
                template_to_question[q_template].append(question)
                complexity_to_template[q_complexity].append(q_template) #so we can look at the most simple cases
                template_to_complexity[q_template].append(q_complexity)
                sparql_to_template[sparql_pattern].append(q_template)
                complexity_to_mod_entities[q_complexity].append(questionModEntities)
                complexity_to_brackets[q_complexity].append(brackets) 
            except Exception as e:
                num_skipped += 1
    print(f"SKIPPED: {num_skipped} / {total_entries}")


load_data()

