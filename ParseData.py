from slippi import Game
import os

files = os.listdir('./slippifile')

file_string = ""

for filename in files:
    game = Game("./slippifile/" + filename)
    frames = game.frames

    char1 = str(frames[0].ports[0].leader.post.character)
    char1 = char1[16:len(char1)]
    char2 = str(frames[0].ports[1].leader.post.character)
    char2 = char2[16:len(char2)]

    char1states = [0 for i in range(383)]
    char2states = [0 for i in range(383)]

    char1string = str(char1)
    char2string = str(char2)
    for i in range(len(frames)):
        char1states[frames[i].ports[0].leader.post.state.value] += 1
        char2states[frames[i].ports[1].leader.post.state.value] += 1

    for i in range(len(char1states)):
        char1string += "," + str(char1states[i])
        char2string += "," + str(char2states[i])

    file_string += char1string + "\n"
    file_string += char2string + "\n"

file = open("games.txt", "w")
test = file.write(file_string)
file.close()