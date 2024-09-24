# ARCWelder

## Vision

The aim here is to use https://github.com/fchollet/ARC-AGI to demonstrate that a suitably designed agentic system with a "Low IQ" LLM should be able to exhibit "High IQ".

I think ARC is a good choice, as it strips the necessary agentic machinery to a minimum.


# What is ARCWelder?

ARCWelder is a DSL for ARC puzzles.

It's different from any other DSL I'm aware of, in that it's designed to facilitate probing and manipulating puzzle-grids, not just coding solutions.

You can think of it as a wrapper around numpy (and some scipy.image tools).

Also provided are 30 or so hand-solved puzzles, as .ipynb notebooks.

The punt here is that a solid agentic system should be pretty good at generating cell k+1 given cells 1...k, and in so doing, solve the puzzle.

My vision of a solver agent is that the agent is at every moment seeking to flesh out what I term as an "Information Frontier". I think that's something bio-brains do.

In each cycle, information gleaned thus far is consolidated, and the agent is free to write arbitrary Python code.
- It might want to probe the puzzle (e.g. list all blue regions together with their bounding-boxes).
- It might want to test out a hypothesis (e.g. "Each red square is given a green border in the output").
- It might want to simplify the puzzle (e.g. rotate each pair such that the largest region in X sits on the bottom edge)

The idea is that it doesn't directly try to "solve" the puzzle. It simply tries to describe what's going on in the simplest way possible.

## Example 1 -- whittling the Information Frontier

Consider a rather contrived / simplistic ARC-style puzzle, wherein the input contains a square of some colour, and the transformation adds diagonal streamers outwards from the corners, and then removes the original.

An initial probe could determine that the input is always a single monochromatic region, and the output comprises four diagonally-connected regions.

Further probing would reveal that the input is a square region, and that the output regions are diagonal lines, extending to off-grid.

Now at this point we can represent the input by four coordinates (TL -> BR), and the output by four {coordinate + direction} pairs (e.g. "Starts at 3, 2 and extends North West off-grid")

Examining this information further, we may observe within this new "Information Frontier" a relation between the input and output information for a given grid; that if we enlarge the input square by 1, the four corners correspond to the four output coordinates.

This may require a little tree-searching and hypothesis testing.

And in this manner we can whittle our information frontier down to ultimately discovering the mind of the puzzle creator.

## Example 2 -- simplification

Imagine a similar puzzle where we start with "triangular"s -- formed by taking a square and removing the portion above or below the leading or anti diagonal. So 4 orientations are possible. And suppose we have to imagine completing the square, and then send a streamer from the "missing" corner to off-grid. One pixel beyond the "missing" corner, if we want to be precise.

Now the agent should quickly figure out that each input is a rotated "triangular", and so can un-rotate each pair so that it is dealing with an "identity triangular" in each case. Then it can proceed with solving this simplified puzzle.

## Example 3 -- generalization

By seeking the simplest explanation, we also arrive at the explanation with most generalizable power. You might imagine in the previons examples that the initial square (or triangle) is not entirely in-grid. But once we have expressed the generative logic in language, it will be easy to code a solver that accounts for such "edge cases".


# Notes

## .ipynb

I've decided to use .ipynb for each puzzle. It's a clean (JSON) format that an LLM should have no difficulty with.

## Unicode

I've also decided to use unicode, as it lets the human (me) see what's going on very clearly. I've since come to question this decision, as unicode is gnarly. I've had to change my VSCode fonts to get everything to render (on my mac), hence I supply a .vscode/ folder. And unfortunately some unicodes won't render the same on GitHub as they do locally on my mac. And Windows users may discover rendering issues.

Pasting unicodes from MacOS Emoji Palette is fail. Don't do this! Some of them have invisible added characters, and I had to write helper code to clean up this repo, and copy-paste from palette.md


# Sponsorship

This work would not have been possible without the kind support of the following organizations and individuals:
- https://www.naptha.ai/
- Alexey (https://github.com/xl0)
- Chris (https://catid.io/)


# About the author (pi)

I led AutoGPT for its first three months in spring/summer 2023, before changing tack and digging into AI theory.
Together with Bojan, I've formed a Discord for AI R&D (you can click thru to the GitHub org and find a link).
You are most welcome to ping me.

I'm most interested in architectures. My feeling is that while Transformers are amazing, we're still missing something that bio-brains possess.

I also find myself pondering the demarcation between computation and awareness. In a meditative state, one is aware. One is one with awareness itself -- the self-aware emptiness of existence, that from which form arises. Now an AI looks like a duck and quacks like a duck. Yet it is not a duck. Where is the essential difference? What would it mean for an AI to be aware?