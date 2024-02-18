import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { NotebookActions, NotebookPanel } from '@jupyterlab/notebook';
import { ICommandPalette } from '@jupyterlab/apputils';
import { ISettingRegistry } from '@jupyterlab/settingregistry';

/**
 * Initialization data for the jupyter_ai extension.
 */
export const biomePlugin: JupyterFrontEndPlugin<void> = {
  id: 'jupyter_ai:magic_ks',
  autoStart: true,
  optional: [],
  requires: [ICommandPalette, ISettingRegistry],
  activate: async (app: JupyterFrontEnd, palette: ICommandPalette) => {
    console.log('Biome AI plugin is activated!');
    const runBiomeCommandID = 'runmagic:biome';

    // Add the new command for the biome magic
    app.commands.addCommand(runBiomeCommandID, {
      label: 'Run Biome Magic',
      execute: () => {
        console.log('Running Biome Magic');
        const current = app.shell.currentWidget;
        if (current && current instanceof NotebookPanel) {
          const notebook = current.content;
          const activeCell = notebook.activeCell;
          if (activeCell && activeCell.model.type === 'code') {
            // Check if the editor is not null
            const editor = activeCell.editor;
            if (editor) {
              // Access the text value safely
              const currentCode = editor.model.sharedModel.getSource();
              console.log('Current code:', currentCode);
              const magicCode = `%%biome chatgpt\n${currentCode}`;
              editor.model.sharedModel.setSource(magicCode);
              // Save the notebook and run the cell
              current.context.save().then(() => {
                NotebookActions.run(notebook, current.sessionContext);
              });
            }
          }
        }
      }
    });

    // Add the keyboard shortcut for the command
    app.commands.addKeyBinding({
      command: runBiomeCommandID,
      args: {},
      keys: ['Accel Shift Enter'],
      selector: '.jp-Notebook.jp-mod-editMode'
    });
    console.log('Adding command to palette');
    palette.addItem({
      command: runBiomeCommandID,
      category: 'Extension Commands',
      rank: 0
    });
  }
};
