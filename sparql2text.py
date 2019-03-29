import json
from argparse import ArgumentParser, Namespace

from pyparsing import ParseException
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.parserutils import prettify_parsetree
from wikidata.client import Client


def main(args: Namespace):
    data = load(args.json)
    client = Client()
    entity = client.get('Q20145', load=True)
    print(entity.description)
    image_prop = client.get('P18')
    image = entity[image_prop]
    for i, entry in enumerate(data, start=1):
        try:
            query = parseQuery(entry["query"])
        except ParseException:
            continue
        print(f"{i:>{len(str(len(data)))}}/{len(data)}")
        print(prettify_parsetree(query))


def load(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    argparser = ArgumentParser(description="Generate natural language text from SPARQL queries using templates.")
    argparser.add_argument("json", help="Input JSON file, "
                                        "e.g. wiki-sparql.json from https://github.com/coastalcph/wiki-sparql")
    main(argparser.parse_args())
