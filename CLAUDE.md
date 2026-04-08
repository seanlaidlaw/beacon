# Beacon Project Guidelines

## README Maintenance

The README.md file should be kept up-to-date with significant changes to the Beacon application functionality. When you make changes that affect:

1. **Command-line interface** (new commands, changed arguments, deprecated features)
2. **Core features** (search, indexing, MCP integration, etc.)
3. **Installation or setup process**
4. **Website content or structure**

Update the README accordingly. The README serves as both project documentation and source for the website content.

## Website Screenshot Generation

The website includes demo outputs from Beacon commands. To keep these current:

1. **Run the screenshot generation script** after significant CLI changes:
   ```bash
   python scripts/generate-screenshots.py
   ```

2. **The script will**:
   - Execute Beacon commands with sample queries on the current codebase
   - Capture terminal output as HTML code blocks
   - Generate HTML snippets in `public/generated/`
   - Create a complete demo section (`demo-section.html`) with:
     - Hero code window showing real `beacon search` output
     - Interactive ask demo showing `beacon ask --help` output

3. **To update the website**:
   - Copy the content from `public/generated/demo-section.html`
   - Replace the hero code-window and CLI demo sections in `public/index.html`
   - Review the generated outputs and adjust formatting if needed

4. **Manual updates** may still be needed for:
   - Feature descriptions
   - Installation steps
   - Command examples in the usage section

## Color Theme Consistency

The Beacon branding uses a "beacon of light" theme with blue/white gradients:

- Primary blue: `#00d4ff`
- Secondary blue: `#80d0ff`
- Accent green: `#27ca3f` (for success/positive indicators)
- Dark background: `#0a0a0a`
- Light text: `#f0f0f0`

All CSS gradients and color choices should align with this palette. Avoid purple/pink colors (#6a5af9, #ff0080, #ff8a00) which conflict with the beacon theme.

## Cursor Animation

The cursor follower should be subtle and responsive:
- Speed: 0.3 (as currently set in script.js)
- Size: 12px with blue glow
- No distracting lag or large glowing dots

## Development Workflow

1. **Make functional changes** to Beacon
2. **Update README.md** if functionality changed
3. **Run screenshot script** to update website demos
4. **Test website locally** with `python serve-website.py`
5. **Commit changes** with descriptive messages

## Website Structure

The website is a static site in `public/`:
- `index.html` - Main page
- `css/style.css` - Styling (keep gradients consistent)
- `js/script.js` - Interactions (cursor, animations)
- `favicon.svg` - Beacon tower logo with light rays

The website showcases Beacon's capabilities with interactive demos and should reflect the current state of the tool.