import json
import os
import json


path = './data/input/synthetic/mic'
cam_infos = []
with open(os.path.join(path, "transform_test.json")) as json_file:
    contents = json.load(json_file)
    frames = contents["frames"]
    for idx, frame in enumerate(frames):
        file_path = f"images/test/r_{idx}"
        frames[idx]["file_path"] = file_path
    contents["frames"] = frames
    newjson_file = json.dumps(contents, indent=4, separators=(',', ': '))
new_file = open("transforms_test.json", "w")
new_file.write(newjson_file)
new_file.close()
