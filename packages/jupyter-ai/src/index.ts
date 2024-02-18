import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
  ILayoutRestorer
} from '@jupyterlab/application';

import {
  IWidgetTracker,
  ReactWidget,
  IThemeManager
} from '@jupyterlab/apputils';
import { IDocumentWidget } from '@jupyterlab/docregistry';
import { IGlobalAwareness } from '@jupyter/collaboration';
import type { Awareness } from 'y-protocols/awareness';
import { buildChatSidebar } from './widgets/chat-sidebar';
import { SelectionWatcher } from './selection-watcher';
import { ChatHandler } from './chat_handler';
import { buildErrorWidget } from './widgets/chat-error';
import { completionPlugin } from './completions';
import { statusItemPlugin } from './status';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { NotebookActions, NotebookPanel } from '@jupyterlab/notebook';
import { ICommandPalette } from '@jupyterlab/apputils';

export type DocumentTracker = IWidgetTracker<IDocumentWidget>;

/**
 * Initialization data for the jupyter_ai extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'jupyter_ai:plugin',
  autoStart: true,
  optional: [IGlobalAwareness, ILayoutRestorer, IThemeManager, ICommandPalette],
  requires: [IRenderMimeRegistry],
  activate: async (
    app: JupyterFrontEnd,
    rmRegistry: IRenderMimeRegistry,
    globalAwareness: Awareness | null,
    restorer: ILayoutRestorer | null,
    themeManager: IThemeManager | null,
    palette: ICommandPalette
  ) => {
    /**
     * Initialize selection watcher singleton
     */
    const selectionWatcher = new SelectionWatcher(app.shell);

    /**
     * Initialize chat handler, open WS connection
     */
    const chatHandler = new ChatHandler();

    let chatWidget: ReactWidget | null = null;
    try {
      await chatHandler.initialize();
      chatWidget = buildChatSidebar(
        selectionWatcher,
        chatHandler,
        globalAwareness,
        themeManager,
        rmRegistry
      );
    } catch (e) {
      chatWidget = buildErrorWidget(themeManager);
    }

    /**
     * Add Chat widget to right sidebar
     */
    app.shell.add(chatWidget, 'left', { rank: 2000 });

    if (restorer) {
      restorer.add(chatWidget, 'jupyter-ai-chat');
    }

    //palette must be initialized
    console.log('Adding command to palette');
    palette.addItem({
      command: 'runmagic:biome',
      category: 'Extension Commands',
      rank: 10
    });

    // Add the new command for the biome magic
    app.commands.addCommand('runmagic:biome', {
      label: 'Run Biome Magic',
      execute: () => {
        console.log('Running Biome Magic');
        const current = app.shell.currentWidget;
        if (current && current instanceof NotebookPanel) {
          const notebook = current.content;
          const activeCell = notebook.activeCell;
          if (activeCell && activeCell.model.type === 'code') {
            const editor = activeCell.editor;
            const currentCode = editor.model.value.text;
            const magicCode = `%%biome\n${currentCode}`;
            editor.model.value.text = magicCode;
            notebook.context.save().then(() => {
              NotebookActions.run(notebook, notebook.sessionContext);
            });
          }
        }
      }
    });

    // Add the keyboard shortcut for the command
    app.commands.addKeyBinding({
      command: 'runmagic:biome',
      keys: ['Accel Shift Enter'],
      selector: '.jp-Notebook.jp-mod-editMode'
    });
  }
};

export default [plugin, statusItemPlugin, completionPlugin];
