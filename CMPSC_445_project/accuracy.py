import pandas as pd
df=pd.read_csv('testset15.csv')
aa=0
ab=0
ac=0
ad=0
ba=0
bb=0
bc=0
bd=0
ca=0
cb=0
cc=0
cd=0
da=0
db=0
dc=0
dd=0
accuracy=0
for i in range(0,43000):

    if df['chosen_class'].loc[i]=='a':
       if df['actual_class'].loc[i]=='a':
            aa+=1
       elif df['actual_class'].loc[i]=='b':
           ab+=1
       elif df['actual_class'].loc[i]=='c':
            ac+=1
       else:
            ad+=1
    elif df['chosen_class'].loc[i]=='b':
       if df['actual_class'].loc[i]=='a':
            ba+=1
       elif df['actual_class'].loc[i]=='b':
           bb+=1
       elif df['actual_class'].loc[i]=='c':
            bc+=1
       else:
            bd+=1
    elif df['chosen_class'].loc[i]=='c':
       if df['actual_class'].loc[i]=='a':
            ca+=1
       elif df['actual_class'].loc[i]=='b':
           cb+=1
       elif df['actual_class'].loc[i]=='c':
            cc+=1
       else:
            cd+=1
    else:
        if df['actual_class'].loc[i] == 'a':
            da += 1
        elif df['actual_class'].loc[i] == 'b':
            db += 1
        elif df['actual_class'].loc[i] == 'c':
            dc += 1
        else:
            dd += 1

matrix=[[aa,ab,ac],[ba,bb,bc],[ca,cb,cc]]
pa=aa/(aa+ab+ac+ad)
pb=bb/(ba+bb+bc+bd)
pc=cc/(ca+cb+cc+cd)
#pd=dd/(da+db+dc+dd)
ra=aa/(aa+ba+ca+da)
rb=bb/(ab+bb+cb+db)
rc=cc/(ac+bc+cc+dc)
#rd=dd/(ad+bd+cd+dd)
#print(matrix)
for row in matrix:
    print(row)
    #print('Accuracy'+str(accuracy/(i+1)))
print("Class A F-Measure: "+str((2*pa*ra)/(ra+pa)))
print("Class B F-Measure: "+str((2*pb*rb)/(rb+pb)))
print("Class C F-Measure: "+str((2*pc*rc)/(rc+pc)))
#print((2*pd*rd)/(rd+pd))
print("Accuracy: "+str((aa+bb+cc+dd)/43000))
classA=(aa+ba+ca+da)
classB=(ab+bb+cb+db)
classC=(ac+bc+cc+dc)
print("A-Test: "+str(classA))
print("B-Test: "+str(classB))
print("C-Test: "+str(classC))

df2=pd.read_csv('trainset12.csv')
print("")

classA=0
classB=0
classC=0
for i in range(0,176595):
    if df2['total_sales_price'].loc[i]<=1399:
        classA+=1
    elif df2['total_sales_price'].loc[i]<=8299:
        classB+=1
    else:
        classC+=1
print("A-Train: "+str(classA))
print("B-Train: "+str(classB))
print("C-Train: "+str(classC))



