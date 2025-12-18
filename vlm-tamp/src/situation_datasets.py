import json
with open("make_coffee.json") as f:
    annotation = json.load(f)
print (annotation)
for step in annotation:
    if step == "000":
        annotation[step]['situations'] = ['The coffee cup is broken.', 'The coffee cup is dirty.', 'The coffee cup is on the floor.', 'There is no coffee cups.']
    elif step == "001":
        annotation[step]['situations'] = ['The coffee maker is missing.']
    elif step == "002":
        annotation[step]['situations'] = ['Ants have gotten into the coffee maker so you can not make coffee.', 'The coffee maker has been sealed shut.', 'The lid of coffee maker is jammed.']
    elif step == "003":
        annotation[step]['situations'] = ['The coffee pod is bad (expired).', 'There is no coffee pod.']
    elif step == "004":
        annotation[step]['situations'] = ['The coffee pod is dropped out onto the table.', 'The coffee pod spills on the floor.']
    elif step == "005":
        annotation[step]['situations'] = ['The coffee maker is missing.']
    elif step == "006":
        annotation[step]['situations'] = ['The coffee pod is dropped out onto the table.', 'The coffee pod spills on the floor.']
    elif step == "007":
        annotation[step]['situations'] = ['The coffee maker has been sealed shut.', 'The lid of coffee maker is jammed.']
    elif step == "008":
        annotation[step]['situations'] = ["The coffee maker has no power.", "the coffee is too hot to drink.", 'The coffee has spilled.', 'The coffee maker cannot be turned on.', 'The coffee maker is not working.', 'The coffee maker switch is stuck.', 'The coffee is overflowing.']
print (annotation)
with open("situations.json", 'w') as f:
    json.dump(annotation, f, indent=4)