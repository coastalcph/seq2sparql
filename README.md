# seq2sparql

In this project, we aim to generate SPARQL queries from natural language queries in the form of noun phrases (e.g., "impressionist painters", "American poets from the 18th century”), so that they can be executed against [Wikidata](https://www.wikidata.org).
Since no large-scale evaluation of semantic parsing for Wikidata queries has been done, we create a resource consisting of:

1. **Parallel datasets.** Natural language queries, along with their SPARQL form and the corresponding results.
    1. Scraped from online tutorials: [wiki-sparql](https://github.com/coastalcph/wiki-sparql).
    2. Using human annotation.
1. **Template-constructed datasets.** Using templates, we will generate natural language queries for SPARQL queries from publically available Wikidata query logs. 
1. **Separate datasets.** We will also build separate (i.e., non-parallel) SPARQL and natural language query (Wikipedia category titles, along with their members, e.g., 18th-century American poets) datasets to enable language modelling.

To give a good indication of the broad-coverage support of the evaluated systems, our evaluation resource focuses on ambiguous cases  (e.g., intransitive and transitive verbs or PP-attachment ambiguities).

Our proposed system to perform semantic parsing on the newly developed dataset involves the following steps:

1. Learn embeddings for words (natural language) and for SPARQL keywords, entities and relations, using word embedding and knowledge graph embedding methods.
1. Induce shared embeddings between natural language and SPARQL, using the Wikidata entity and relation labels as training data.
1. Train on the annotated parallel data to generate SPARQL from text and vice versa.
1. Learn language models for natural language and for SPARQL queries.
1. Back-translation: use the trained model to generate SPARQL from text and vice versa, and use the result as training data for the other direction.
1. Fine-tune the model by learning from denotations with policy gradient.

As baselines, we develop a rule-based system, as well as an information retrieval system based on the Wikipedia API.

Notes:
1. Use transformers==4.15.0 for training t5 models
2. Use transformers==4.6.0 for training other seq2seq models