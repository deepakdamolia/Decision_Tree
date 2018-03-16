# Decision_Tree

Simple implementation of Decision Tree and Pruning. Has working examples on two datasets for demonstration.
All the scripts are in python 3.5

## DESCRIPTION

decision_tree.py: It contains implementation of Decision Tree using "Information Gain" as the selection criterion for the best attribute.
decision_tree_gain_ratio.py: It contains implementation of Decision Tree using "Gain Ratio" as the selection criterion.
data_manip.py: It contains functions to create data partitions, calculate accuracy etc.
q1_run_decision_tree_on_lenses: Answer to question 1 (Please refer [Decision_tree_problems](Decision_tree_problems.pdf))
q2_run_decision_tree_on_other_dataset.py : Answer to question 2 (Please refer [Decision_tree_problems](Decision_tree_problems.pdf))
run_decision_tree_on_other_dataset_basic_demo.py: Run it to get an idea of the inner procedures.
run_decision_tree_on_other_dataset_using_gain_ratio.py : Answer to question 2 using gain ratio as selection rule


## INSTRUCTIONS
1. Keep all the files in same folder (Do not move any of them, as modules are interdependent.)
2. Run q1_run_decision_tree_on_lenses.py for question 1:
   *5-fold cross validation has been used.
   *Node_id: "{2}_{3} Tear_production" means a the node is the 3rd node at level 2 and has best attribute = "Tear_production"
3. Run q2_run_decision_tree_on_other_dataset.py for Ques. 2:
  *L and K values used are L = [15,20] and K = [15,20,30,40,50] to get 10 combinations
4. Run run_decision_tree_on_other_dataset_using_gain_ratio.py for solving Ques.2 using gain ratio based Decision Tree.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
