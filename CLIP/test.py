with open("aa.txt","r") as f:
    lines=f.readlines()
    c=0
    for line in lines:
        line=eval(line)
        c+=len(line)
    print(c)