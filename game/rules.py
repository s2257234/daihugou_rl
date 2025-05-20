#-------ゲームのルール・判定ロジック-------

def compare_cards(card1, card2, revolution=False):
    order = get_card_strength(card1, revolution)
    return order - get_card_strength(card2, revolution)

def get_card_strength(card, revolution):
    if card.suit == "JOKER":
        return 14 if not revolution else -1
    base = card.rank if card.rank != 1 else 14  # A=14, 2=15
    return 15 - base if revolution else base

def is_valid_play(play, field_cards, revolution):
    if not field_cards:
        return True  # 場が空
    if len(play) != len(field_cards):
        return False
    return all(compare_cards(p, f, revolution) > 0 for p, f in zip(play, field_cards))
