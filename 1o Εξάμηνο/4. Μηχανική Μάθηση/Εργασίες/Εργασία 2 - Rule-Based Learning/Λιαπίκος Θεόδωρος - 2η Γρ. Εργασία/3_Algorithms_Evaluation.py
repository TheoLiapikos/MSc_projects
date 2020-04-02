# Μαζική δοκιμή παραμέτρων
import Orange

wineData = Orange.data.Table('wine')

for c in (
#        Orange.classification.rules.CN2Learner(),
        Orange.classification.rules.CN2UnorderedLearner(),
        ):
    
    learner = c
    
    # Δήλωση evaluator = Entropy
    learner.rule_finder.quality_evaluator = Orange.classification.rules.EntropyEvaluator()
    
    # Δήλωση evaluator = Laplace
#    learner.rule_finder.quality_evaluator = Orange.classification.rules.LaplaceAccuracyEvaluator()
    
    print("\n\nMethod:")
    print(c)
    results_total= []
    for bm in (2,3,4,5,6):
        for mce in (7,9,11,13,15):
            for mrl in (2,3,4,5,6):
                results = []
                results.append(bm)
                results.append(mce)
                results.append(mrl)
                learner.rule_finder.search_algorithm.beam_width = bm
                learner.rule_finder.general_validator.min_covered_examples = mce
                learner.rule_finder.general_validator.max_rule_length = mrl
                results_model = Orange.evaluation.testing.CrossValidation(wineData, [learner], k=10,random_state=42)
                acc = Orange.evaluation.scoring.Precision(results_model, average='macro')
                results.append(acc)
                rc = Orange.evaluation.scoring.Recall(results_model, average='macro')
                results.append(rc)
                f1 = Orange.evaluation.scoring.F1(results_model, average='macro')
                results.append(f1)
                results_total.append(results)
        
    # Εκτύπωση των συνολικων αποτελεσμάτων
    print('bm\t', 'mce\t', 'mrl\t', 'Prec\t', 'Recall\t', 'F1\t')
    for result in results_total:
        print('{}\t {}\t {}\t {:.3f}\t {:.3f}\t {:.3f}\t'.format
              (result[0], result[1], result[2], result[3][0], result[4][0], result[5][0]))


