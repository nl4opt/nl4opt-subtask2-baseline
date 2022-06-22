import sys
sys.path.append('.')

import numpy as np
import parsers
import jsonlines
import scoring


def similarity_test():
    parser = parsers.Parser()

    assert parser.similarity('local smelly fish', 'smelly fish') > parser.similarity('local smelly fish', 'jelly fish')
    assert parser.similarity('smelly local fish', 'smelly fish') > parser.similarity('smelly local fish', 'jelly fish')
    assert parser.similarity('color printer', 'colored printer') > parser.similarity('color printer', 'printer')

    print('Similarity metric test passed')


def scoring_test():


    json_parser = parsers.JSONFormulationParser()
    xml_parser = parsers.ModelOutputXMLParser()
    # regular test pipeline of a few examples
    with jsonlines.open('notebooks/demo_examples/annotated.jsonl') as reader:
        grounds = []
        xml_outputs = []

        for i, line in enumerate(reader.iter()):
            ground = json_parser.parse(line)
            grounds.append(ground)
            xml_outputs.append(xml_parser.parse_file(f'notebooks/demo_examples/modeloutput{i}.txt', ground.entities))


        canonical_1 = [parsers.convert_to_canonical(p1) for p1 in grounds]
        canonical_2 = [parsers.convert_to_canonical(p2) for p2 in xml_outputs]

        
        scores = [scoring.per_example_scores(canonical_1[i].objective,
                                             canonical_1[i].constraints,
                                             canonical_2[i].objective,
                                             canonical_2[i].constraints) for i in range(2)]

        # correct example
        assert scores[0] == (0, 0, 3)
        # 1 fp
        assert scores[1] == (1, 0, 4)

        print('Accuracy: ', scoring.overall_score(
            [p1.objective for p1 in canonical_1],
            [p1.constraints for p1 in canonical_1],
            [p2.objective for p2 in canonical_2],
            [p2.constraints for p2 in canonical_2]
        ))

    # all correct
    score = scoring.per_example_scores(
        np.array([1, 1]),
        np.array([[1, 1, 1], [1, 0, 4]]),
        np.array([1, 1]),
        np.array([[1, 1, 1], [1, 0, 4]]),
    )
    assert score == (0, 0, 3)

    # all false positives
    score = scoring.per_example_scores(
        np.array([1, 1]),
        np.array([[1, 1, 1], [1, 0, 4]]),
        np.array([4, 1]),
        np.array([[1, 0, 1], [1, 0, 2]]),
    )
    assert score == (3, 0, 3)

    # all false positives with more predictions than ground truth
    score = scoring.per_example_scores(
        np.array([1, 1]),
        np.array([[1, 1, 1], [1, 0, 4], [3, -3, 7]]),
        np.array([4, 1]),
        np.array([[1, 0, 1], [1, 0, 2]]),
    )
    assert score == (3, 0, 3)

    # duplicates
    # objective is correct, 1 duplicate corrrect constraint, 1 duplicate wrong constraint
    score = scoring.per_example_scores(
        np.array([1, 1]),
        np.array([[1, 1, 1], [1, 1, 1], [1, 0, 4], [1, 0, 4]]),
        np.array([1, 1]),
        np.array([[1, 1, 1], [1, 0, 2]]),
    )
    assert score == (1, 0, 3)

    # more predictions than ground truth
    # should have 1 fn
    score = scoring.per_example_scores(
        np.array([1, 1]),
        np.array([[1, 1, 1], [1, 0, 4], [2, 3, 4]]),
        np.array([1, 1]),
        np.array([[1, 1, 1], [1, 0, 4]]),
    )
    assert score == (1, 0, 3)    

    # fewer predictions and one wrong constraint = 1 fp and 2 fn
    score = scoring.per_example_scores(
        np.array([1, 1]),
        np.array([[1, 1, 0]]),
        np.array([1, 1]),
        np.array([[1, 1, 1], [1, 0, 4], [2, 3, 4]]),
    )
    assert score == (1, 2, 4)

    # fewer predictions and all wrong
    score = scoring.per_example_scores(
        np.array([1, 3]),
        np.array([[1, 1, 0]]),
        np.array([1, 1]),
        np.array([[1, 1, 1], [1, 0, 4], [2, 3, 4]]),
    )
    assert score == (2, 2, 4)

    # contains all correct answers but still has too many predictions
    score = scoring.per_example_scores(
        np.array([1, 1]),
        np.array([[1, 1, 1], [1, 0, 4], [2, 3, 4], [3, 1, 9], [1, -1, 1], [4, 3, -2], [1, 1, -1]]),
        np.array([1, 1]),
        np.array([[1, 1, 1], [1, 0, 4], [2, 3, 4]]),
    )
    assert score == (4, 0, 4)
    print('Scoring test passed')


def json_parsing_test(fname):
    parser = parsers.JSONFormulationParser()
    with jsonlines.open(fname) as reader:
        examples = [line for line in reader.iter()]
        parsed = [parser.parse(ex) for ex in examples]
        # parsed = [ex for ex in parsed if ex is not None]

        for i in parsed:
            for j in parsed:
                canonical_i = parsers.convert_to_canonical(i)
                canonical_j = parsers.convert_to_canonical(j)
                fp, fn, d = scoring.per_example_scores(canonical_i.objective, canonical_i.constraints,
                                                       canonical_j.objective,
                                                       canonical_j.constraints)
    print(f'Parsed JSON file: {fname}')

def xml_parsing_test():
    parser = parsers.ModelOutputXMLParser()
    parsed = [parser.parse_file(f'examples/example{i}.txt') for i in range(1,9)]

    for i in parsed:
        for j in parsed:
            canonical_i = parsers.convert_to_canonical(i)
            canonical_j = parsers.convert_to_canonical(j)
            fp, fn, d = scoring.per_example_scores(canonical_i.objective, canonical_i.constraints,
                                                   canonical_j.objective,
                                                   canonical_j.constraints)

    print('Parsed XML files')

if __name__ == "__main__":
    similarity_test()
    scoring_test()
    xml_parsing_test()
    json_parsing_test('examples/datasets/order_mapping/train.jsonl')
    json_parsing_test('examples/datasets/order_mapping/dev.jsonl')
    json_parsing_test('examples/datasets/order_mapping/test.jsonl')

    # with jsonlines.open('examples/test_ex.jsonl') as reader:
    #     examples = [line for line in reader.iter()]
    #     parsed = [parser.parse_dict(ex) for ex in examples]
    #
