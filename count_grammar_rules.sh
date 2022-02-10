#!/bin/bash

tr '\n' ' ' < cfq/dataset.json | grep -oP '(?<="stringValue": ")[^"]*(?=", "type": "GRAMMAR_RULE")' | sort -u
