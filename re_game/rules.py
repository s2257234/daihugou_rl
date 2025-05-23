# カードが現在の場に対して有効な出し方かどうかを判定する関数
def is_valid_move(card, current):
    # 場にカードがない（リセットされている）場合は何でも出せる
    if not current:
        return True
    
    # 出そうとしているカードがジョーカーなら必ず出せる
    if card.is_joker:
        return True

    # 現在の場のカードがジョーカーなら,出せない
    if current.is_joker:
        return False 

    # 通常は、場のカードのランク以上でないと出せない
    return card.rank > current.rank
