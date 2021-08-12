def lat_acceleration(v,r):
    # Your code here.
    a = v**2/r
    return a
    
veloctity = 30 #m/s
radius = 90 #m

acc = lat_acceleration(veloctity, radius)

print(acc)
