import os
import re
import argparse


def expand_includes(shader, input_dir):
    """
    Replace #include "file" lines with the contents of that file so the
    runtime compiler (dxcompiler.dll) needs no include handler.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.join(input_dir, fname)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            included_code = f.read()
        return expand_includes(included_code, input_dir)

    return include_pattern.sub(replacer, shader)


def raw_delim(shader_code):
    delim = "hlsl"
    while f"){delim}\"" in shader_code:
        delim += "_x"
    return delim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--table", action="store_true", help="also emit a name to source table")
    args = parser.parse_args()

    names = []
    with open(args.output_file, "w", encoding="utf-8") as out:
        out.write("// Auto-generated shader embedding\n\n")
        for fname in sorted(os.listdir(args.input_dir)):
            if fname.endswith(".hlsl"):
                shader_path = os.path.join(args.input_dir, fname)
                shader_name = fname.replace(".hlsl", "")
                names.append(shader_name)
                with open(shader_path, "r", encoding="utf-8") as f:
                    shader_code = expand_includes(f.read(), args.input_dir)
                delim = raw_delim(shader_code)
                out.write(f'static const char * hlsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
        if args.table:
            out.write("static const struct { const char * name; const char * source; } hlsl_table[] = {\n")
            for n in names:
                out.write(f'    {{ "{n}", hlsl_{n} }},\n')
            out.write("};\n")


if __name__ == "__main__":
    main()
