import matplotlib.pyplot as plt
import os


cloth = "./dataset/Cloth mask"
n95 = "./dataset/N95 Mask"
no = "./dataset/No Mask"
surgical = "./dataset/Surgical Mask"

totalClothFiles = 0
totalN95Files = 0
totalNoFiles = 0
totalSurgicalFiles = 0

for base, dirs, files in os.walk(cloth):
  for Files in files:
    totalClothFiles += 1

for base, dirs, files in os.walk(n95):
  for Files in files:
    totalN95Files += 1

for base, dirs, files in os.walk(no):
  for Files in files:
    totalNoFiles += 1

for base, dirs, files in os.walk(surgical):
  for Files in files:
    totalSurgicalFiles += 1

x = range(4)

x_labels = ["Cloth", "N95", "No", "Surgical"]

y = [totalClothFiles, totalN95Files, totalNoFiles, totalSurgicalFiles]

plt.bar(x,y, color=['black', 'red', 'blue', 'green'])

plt.title("Number of masks per type")
plt.ylabel("Total number of masks per type", fontsize =10)
plt.xticks(x, x_labels)


for index, data in enumerate(y):
   plt.text(x=index, y=data+1, s=f"{data}",
             fontdict=dict(fontsize=12, color='maroon'))

# plt.savefig("MaskDistribution.pdf")



