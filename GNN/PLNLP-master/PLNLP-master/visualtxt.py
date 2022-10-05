import matplotlib
from matplotlib import pyplot as plt
import argparse
import os

def readHit(path,dict):
    """读取Hit格式文件"""
    with open(path,"r") as f: 
        lines=f.readlines()
        use_lines=[]
        for idx,line in enumerate(lines):
            if line[:4]=="Run:":
                use_lines.append((lines[idx-1],line))
        lines_list=[]
        for use_line in use_lines:
            lines_list.append((use_line[0][5:].strip(),use_line[1].split(sep=",")))
        for line_list in lines_list:
            tmp_list=[]
            for it in line_list[1]:
                tmp_list.append([word.strip() for word in it.split(":")])
            dict[line_list[0]+"."+tmp_list[0][1]+"."+tmp_list[1][1]]=[
                float(num[1].replace("%","")) for num in tmp_list[2:]]  

def makepltdata(lines_dict,Hit,Run,num_x,idx):
    ydata=[lines_dict[Hit+"."+Run+"."+str(5*i).zfill(2)][idx] for i in range(1,num_x+1)]
    xdata=[5*i for i in range(1,num_x+1)]
    return (xdata,ydata)

def plotabloss(lines_dict,Hit,num_x,idx,a,b):
    name=["Loss","Learning rate","Valid acc","Test acc"]
    fig=plt.figure(figsize=(15,20))
    fig.suptitle("Hit "+Hit+" "+name[idx])
    fig.subplots_adjust(hspace=0.4)
    for i in range(0,a):
        for j in range(0,b):
            plt.subplot(a,b,i*b+j+1)
            xdata,ydata=makepltdata(lines_dict,Hit,str(i*b+j+1).zfill(2),num_x,idx)
            plt.title("Run "+str(i*b+j+1).zfill(2))
            plt.plot(xdata,ydata)
    if ~os.path.exists("./save"+"/Hit"+Hit+"-"+name[idx]):
        os.makedirs("./save"+"/Hit"+Hit+"-"+name[idx])
    plt.savefig("./save"+"/Hit"+Hit+"-"+name[idx]+"/"+name[idx]+".png")

def main():
    parser=argparse.ArgumentParser(description="Demo")
    parser.add_argument("-p","--path")
    parser.add_argument("-e","--epochs",default=100,type=int)
    parser.add_argument("-a",default=5,type=int)
    parser.add_argument("-b",default=2,type=int)
    args=parser.parse_args()
    Hits=["20","50","100"]
    Dict={}

    readHit(args.path,Dict)
    for Hit in Hits:
        for i in range(4):
            plotabloss(Dict,Hit,args.epochs,i,int(args.a),int(args.b))

if __name__ == "__main__":
    main()