def fairRations(B):
    ar=[1 for x in B if x%2==1]
    if sum(ar)!=0:
        count=0
        for i in range(len(B)-1):
            if B[i]%2==1:
                B[i]+=1
                B[i+1]+=1
                count+=2
        for x in B:
            if x%2!=0:
                return 'NO'
        return count
    else:
        return 0
