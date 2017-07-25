import sys
from subprocess import call


def createCommand(videoloc,outdir):
    command = 'ffmpeg -i "'+videoloc+'" -vf fps=1/60 -f image2 "'+outdir+'video-frame%03d.png"'
    return command

if len(sys.argv) >= 4:
    print("DEBUG")
    print(createCommand(sys.argv[1], sys.argv[2]))
else:
    call([createCommand(sys.argv[1], sys.argv[2])], shell=True)