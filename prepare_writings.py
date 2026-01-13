import glob
import sys
import os

def main():
    filenames = ["writings_bk1.txt", "writings_bk2.txt", "writings_bk4.txt"]
    # filenames = ["writings_bk4.txt"]
    chapter = "0"
    book = "Book one"
    par = ""
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("BOOK"):
                    book = line
                    # print(book)
                elif line.startswith("CHAPTER"):
                    chapter = line + "/"
                    line = next(f).strip()
                    chapter += line
                    # print(chapter)
                elif len(line) < 1:
                    if len(par) > 2:
                        print(book+"/"+chapter, '\t', par)
                    par = ""
                else:
                    if len(line) > 2:
                        par += line + " "

if __name__ == "__main__":
    main()
