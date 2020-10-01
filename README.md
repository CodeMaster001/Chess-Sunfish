
 Sunfish is a simple, but strong chess engine, written in Python, mostly for teaching purposes. Without tables and its simple interface, it takes up just 111 lines of code! Yet it plays at rating 1800-1900 at Lichess. It’s simple open source chess engine under the GPL written by Thomas Dybdahl Ahle in Python for didactic purposes, inspired by Harm Geert Muller's Micro- Max. Sunfish supports the Chess Engine Communication protocol to play with a graphical interface like XBoard or PyChess. 
 The program uses Unicode glyphs to display the pieces within the terminal,making it look a little more chess- like than GNU chess. Moves are inputted by specifying the starting and ending co-ordinates,so the aggressive opening which moves the king's pawn to e4 would be inputted e2e4.Note that this can be slightly longer than the more common algebraic notation(in which the previous move would be written e4),but makes it much easier for computation. In this engine Lower case letters represent black pieces (p,r,n,b,k,q), and upper case letters represent white pieces (P,R,N,B,K,Q).Black-White-Empty matrix(BWE) is solely a listing of strings which represent each square on the board. This matrix is compared with an internally stored matrix within the Chess Engine.This suggests the Chess Engine can understand : 
 
i)where the piece moved from 
 
ii)where it moved to and construct a chess command from it. Moving pawn piece A2 to A4 at the beginning of the game would require command ‘a2a4’.         
Original Sunfish repository

https://github.com/thomasahle/sunfish

This repository add machine learning  support  to sunfish code

The code is buggy ,but still worth of my free time.
