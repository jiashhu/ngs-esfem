from PIL import Image
import os
from Package_MyCode import FO

BaseDir = '/Users/hjs/Desktop/MN3'
flist = FO.Get_File_List(BaseDir)
for fname in flist:
    if fname.endswith('.png'):
        newname = '-'.join(fname.split('.')[:-1])
        rgba = Image.open(os.path.join(BaseDir,fname))
        rgb = Image.new('RGB', rgba.size, (255, 255, 255))  # white background
        rgb.paste(rgba, mask=rgba.split()[3])               # paste using alpha channel as mask
        rgb.save(os.path.join(BaseDir,'{}.pdf'.format(newname)), "PDF",
                            resolution=100.0, save_all=True)

        