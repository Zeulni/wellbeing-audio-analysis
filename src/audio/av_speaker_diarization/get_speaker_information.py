import pickle

with open("./demo/001/pywork/faces.pckl", "rb") as file:
    faces = pickle.load(file)

with open("./demo/001/pywork/scene.pckl", "rb") as file:
    scene = pickle.load(file)
    
with open("./demo/001/pywork/scores.pckl", "rb") as file:
    scores = pickle.load(file)
    
with open("./demo/001/pywork/tracks.pckl", "rb") as file:
    tracks = pickle.load(file)

print(faces)