from __future__ import annotations

from functools import lru_cache
from typing import List

from game.card import Card, SUITS, RANKS
from agents.agent_utills.feature_extractor import CardEncoder


def _action_key_from_cards(cards: List[Card]) -> str:
    ce = CardEncoder()
    idxs = []
    for c in cards:
        try:
            idxs.append(int(ce.card_index(c)))
        except Exception:
            continue
    idxs.sort()
    return '|'.join(str(i) for i in idxs)


@lru_cache(maxsize=1)
def canonical_action_keys(include_pass: bool = True) -> List[str]:
    """Return full canonical action vocabulary (deterministic order).

    Actions cover:
      - Singles (52 + Joker)
      - Same-rank sets (pairs/triples/quads), with optional Joker
      - Straights length 3..6 per suit, with optional Joker substitution
      - PASS (optional)

    Keys are sorted card indices joined by '|'.
    """
    keys: List[str] = []
    key_set = set()

    def add(cards: List[Card]):
        k = _action_key_from_cards(cards)
        if k and k not in key_set:
            key_set.add(k)
            keys.append(k)

    # Singles
    for suit in SUITS:
        for rank in RANKS:
            add([Card(suit, rank)])
    joker = Card(is_joker=True)
    add([joker])

    # Same-rank sets (2..4), with optional Joker
    import itertools
    for rank in RANKS:
        rank_cards = [Card(suit, rank) for suit in SUITS]
        for size in range(2, 5):
            for comb in itertools.combinations(rank_cards, size):
                add(list(comb))
            # with Joker
            for comb in itertools.combinations(rank_cards, size - 1):
                add(list(comb) + [joker])

    # Straights length 3..6 per suit, with optional Joker substitution
    suit_cards = {suit: {rank: Card(suit, rank) for rank in RANKS} for suit in SUITS}
    for suit in SUITS:
        for length in range(3, 7):
            for start in range(1, 14):
                ranks = [((start + i - 1) % 13) + 1 for i in range(length)]
                if 2 in ranks and ranks[-1] != 2:
                    continue
                base_cards = [suit_cards[suit][r] for r in ranks]
                add(base_cards)
                # Joker variants (replace one position)
                for j in range(length):
                    cards = [suit_cards[suit][r] for i, r in enumerate(ranks) if i != j]
                    add(cards + [joker])

    if include_pass:
        keys.append('PASS')
    return keys


def action_vocab_size(include_pass: bool = True) -> int:
    return len(canonical_action_keys(include_pass=include_pass))


__all__ = ["canonical_action_keys", "action_vocab_size"]