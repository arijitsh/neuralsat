from beartype import beartype

from setting import Settings

INPUT_SPLIT_RESTART_STRATEGIES = [
    {'input_split': True, 'abstract_method': 'forward+backward', 'decision_method': 'naive', 'decision_topk': 1},
    {'input_split': True, 'abstract_method': 'backward',         'decision_method': 'smart', 'decision_topk': 1},
    {'input_split': True, 'abstract_method': 'crown-optimized',  'decision_method': 'smart', 'decision_topk': 1},
    {'input_split': True, 'abstract_method': 'forward+backward', 'decision_method': 'naive', 'decision_topk': 1}, # for attack
]

HIDDEN_SPLIT_RESTART_STRATEGIES = [
    {'input_split': False, 'abstract_method': 'crown-optimized', 'decision_method': 'smart', 'decision_topk': 10},
    {'input_split': False, 'abstract_method': 'crown-optimized', 'decision_method': 'smart', 'decision_topk': 20},
]

# Diversity restart strategy: used by NeuralSAT-Div between sampling rounds.
# Alternates between smart (FSB) and naive (random) hidden splitting to
# ensure successive DPLL runs explore structurally different activation
# regions — the same effect as HighDiv alternating stochastic CDCL(T) and
# CCSS with different variable initialization partitions.
DIVERSITY_RESTART_STRATEGIES = [
    {'input_split': False, 'abstract_method': 'crown-optimized', 'decision_method': 'smart', 'decision_topk': 10},
    {'input_split': False, 'abstract_method': 'crown-optimized', 'decision_method': 'naive', 'decision_topk': 1},
    {'input_split': False, 'abstract_method': 'forward+backward', 'decision_method': 'naive', 'decision_topk': 1},
    {'input_split': False, 'abstract_method': 'crown-optimized', 'decision_method': 'smart', 'decision_topk': 20},
]


@beartype
def get_restart_strategy(nth_restart: int, input_split: bool = False) -> dict:
    # --- NeuralSAT-Div: rotate through diversity strategies when sampling ---
    if getattr(Settings, 'use_diversity_sampling', False) and not input_split:
        idx = nth_restart % len(DIVERSITY_RESTART_STRATEGIES)
        strategy = DIVERSITY_RESTART_STRATEGIES[idx]
        if Settings.use_save_reasoning_step:
            strategy = dict(strategy)
            strategy['abstract_method'] = 'crown-optimized'
        return strategy

    if input_split:
        if not Settings.use_restart:
            strategy = {'input_split': True, 'abstract_method': Settings.default_abstraction_method, 'decision_method': 'smart', 'decision_topk': 1}
        elif nth_restart >= len(INPUT_SPLIT_RESTART_STRATEGIES):
            strategy = INPUT_SPLIT_RESTART_STRATEGIES[-1]
        else:
            strategy = INPUT_SPLIT_RESTART_STRATEGIES[nth_restart]
    else:
        if nth_restart >= len(HIDDEN_SPLIT_RESTART_STRATEGIES):
            strategy = HIDDEN_SPLIT_RESTART_STRATEGIES[-1]
        else:
            strategy = HIDDEN_SPLIT_RESTART_STRATEGIES[nth_restart]
            
    if Settings.use_save_reasoning_step:
        strategy['abstract_method'] = 'crown-optimized'
    return strategy
    
    
        
    