"""
Goal:
* Define a bond fingerprint that contains the same information as the AM1CCC types

Approach:
* Extract atomic primitives, bond primitives, etc. from the AM1CCC SMIRKS patterns
"""

from timemachine.ff import Forcefield
import lark

bond = lark.Lark("""
    labeled_bond:
        | "[" atom1 "]" bond_primitive "[" atom2 "]"

    atom1: atom_expr ":1"
    atom2: atom_expr ":2"

    atom_expr:
        | atom_primitive ~ 1..3
        | atom_expr connective atom_expr

    atom_primitive: 
        | "#" NUMBER -> element
        | "a"        -> aromatic
        | "X" NUMBER -> connectivity
        | "r" NUMBER -> ring_size
        | "+" NUMBER -> pos_formal_charge
        | "-" NUMBER -> neg_formal_charge

    bond_primitive:
        | "~" -> any_bond
        | "-" -> single_bond
        | "=" -> double_bond
        | "#" -> triple_bond
        | ":" -> aromatic_bond
        | "@" -> ring_bond

    connective:
        | "&" -> and
        | ";" -> and
        | "," -> or

    %import common.NUMBER""", start="labeled_bond")




if __name__ == "__main__":
    ff = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")
    patterns = ff.q_handle.smirks
    print(f"total # patterns: {len(patterns)}")
    print(f"5 example_patterns: {patterns[:5]}\n")
    print(f"single longest pattern: {sorted(patterns, key=len)[-1]}\n")

    # try to parse everything
    passed = []
    failed = []
    for s in patterns:
        try:
            bond.parse(s)
            passed.append(s)
        except:
            failed.append(s)

    # how did we do?
    print(f"{100 * len(passed) / len(patterns):.1f}% successfully parsed")

    # print an example parse tree
    example = passed[0]
    print("\nfor example:\n" + example + "\n-->")
    print(bond.parse(example).pretty())

    print("\n" + "-" * 50 + "\n")

    # categorize failures

    # recursive smarts
    recursive_patterns = [s for s in failed if '$' in s]
    print(f"{100 * len(recursive_patterns) / len(failed):.1f}% of failures contain recursive smarts")
    print('5 examples: ', recursive_patterns[:5])
    print("\n" + "-" * 50 + "\n")

    # other: extended patterns [atom :1](~[atom])(~[atom :2])
    other_patterns = [s for s in failed if '$' not in s]
    print(f"{100 * len(other_patterns) / len(failed):.1f}% of failures do not contain recursive smarts")
    print('all examples: ', other_patterns)
    print("\n" + "-" * 50 + "\n")
