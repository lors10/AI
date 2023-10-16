
import chess

board = chess.Board()
# crea la scacchiera con tutti i pedoni
#
# r | R = rook (torre)
# n | N = knight (cavaliere)
# b | B = bishop (vescovo)
# q | Q = queen (regina)
# k | K = king (re)
# p | P = pawn (pedone)

# come è raffigurata una scacchiera
#   a  b  c  d  e  f  g  h
# 8 r  n  b  q  k  b  n  r 8
# 7 p  p  p  p  p  p  p  p 7
# 6 .  .  .  .  .  .  .  . 6
# 5 .  .  .  .  .  .  .  . 5
# 4 .  .  .  .  .  .  .  . 4
# 3 .  .  .  .  .  .  .  . 3
# 2 P  P  P  P  P  P  P  P 2
# 1 R  N  B  Q  K  B  N  R 1
#   a  b  c  d  e  f  g  h

print(board)

#mosse = board.legal_moves

bho = chess.Move.from_uci("a8a1") in board.legal_moves

mossa1 = board.push_san("e4")
print(mossa1)

print(board)

mossa2 = board.push_san("e5")
print(mossa2)

print(board)

mossa3 = board.push_san("Qh5")
print(mossa3)

print(board)

mossa4 = board.push_san("Nc6")
print(mossa4)

print(board)

mossa5 = board.push_san("Bc4")
print(mossa5)

print(board)

mossa6 = board.push_san("Nf6")
print(mossa6)

print(board)

mossa7 = board.push_san("Qxf7")
print(mossa7)

print(board)

