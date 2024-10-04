import os, csv

with open("Gunfiles.csv", 'w') as f:
    x = 100032
    writer = csv.writer(f)
    writer.writerow(["filename","fold","target","category","esc10","src_file","take"])

    for path, dirs, files in os.walk("/home/nimnim/gunshot/gunshot2023summer/datasetsplitsmall/train/ssecondgunshotname"):
        for filename in files:
            x=x+1
            writer.writerow([str(filename), 1 ,0,"Gunshots",True,x, "A"])
    for path, dirs, files in os.walk("/home/nimnim/gunshot/gunshot2023summer/datasetsplitsmall/train/other"):
        for filename in files:
            x=x+1
            writer.writerow([str(filename), 1 ,0,"Other",True,x, "A"])



    for path, dirs, files in os.walk("/home/nimnim/gunshot/gunshot2023summer/datasetsplitsmall/val/ssecondgunshotname"):
        for filename in files:
            x=x+1
            writer.writerow([str(filename), 2 ,0,"Gunshot",True,x, "B"])
    for path, dirs, files in os.walk("/home/nimnim/gunshot/gunshot2023summer/datasetsplitsmall/val/other"):
        for filename in files:
            x=x+1
            writer.writerow([str(filename), 2 ,0,"Other",True,x, "B"])



    for path, dirs, files in os.walk("/home/nimnim/gunshot/gunshot2023summer/datasetsplitsmall/test/ssecondgunshotname"):
        for filename in files:
            x=x+1
            writer.writerow([str(filename), 3 ,0,"Gunshot",True,x, "C"])
    for path, dirs, files in os.walk("/home/nimnim/gunshot/gunshot2023summer/datasetsplitsmall/test/other"):
        for filename in files:
            x=x+1
            writer.writerow([str(filename), 3 ,0,"Other",True,x, "C"])

