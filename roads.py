
def connect(roads,current=-1,seen=set(),warehouse=1):
    maxi=set()
    checked=set()
    maxi.add(0)
    for x in range(len(roads)):
        if(roads[x][0] not in checked and roads[x][0] not in seen and (roads[x][1]==current or current==-1 )):
            cur=roads[x][0]
            seen.add(cur)
        elif(roads[x][1] not in checked and roads[x][1] not in seen and (roads[x][0]==current or current==-1)):
            cur=roads[x][1]
            seen.add(cur)
        else:
            continue
        checked.add(cur)
        if(cur!=warehouse):
            maxi.add(1+connect(roads,cur,seen))
        else:
            maxi.add(connect(roads,cur,seen))
        seen.remove(cur)
    return(max(maxi))

print(connect([(1,2),(1,3),(3,4),(1,5),(5,4),(1,6),(1,7)]))