package com.compomics.deeplc_gui.view;

import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import javax.swing.text.BadLocationException;
import javax.swing.text.Document;
import org.apache.log4j.WriterAppender;
import org.apache.log4j.spi.LoggingEvent;

/**
 * Appender class for writing log messages to a JTextArea.
 *
 * @author Niels Hulstaert
 */
public class LogTextAreaAppender extends WriterAppender implements Runnable {

    /**
     * The log text area to log to.
     */
    private JTextArea logTextArea;
    /**
     * The cycle of string characters to iterate over during waiting.
     */
    private String[] cycle = new String[]{"/", "-", "\\", "|"};
    /**
     * The status boolean to notify if the appender is being written to.
     */
    private boolean lock = false;
    /**
     * The status boolean indicating the appender should perform the cycle.
     */
    private boolean loading = false;
    /**
     * boolean indicating the logger is closed.
     */
    private boolean closed = false;
    /**
     * The animation thread.
     */
    private Thread animationThread;
    /**
     * The current cycling index for an animation.
     */
    private int index = 0;

    /**
     * No-arg constructor.
     */
    public LogTextAreaAppender() {
        animationThread = new Thread(this);
        animationThread.start();
    }

    public void setLogTextArea(JTextArea logTextArea) {
        this.logTextArea = logTextArea;
    }

    @Override
    public void close() {
        closed = true;
        animationThread.interrupt();
    }

    public void setLoading(boolean loading) {
        this.loading = loading;
    }

    @Override
    public void append(LoggingEvent event) {
        while (lock) {
            try {
                Thread.sleep(100);
                //give gui time to handle it
            } catch (InterruptedException ex) {
                //ignore
            }
        }

        final String message = this.layout.format(event);
        SwingUtilities.invokeLater(() -> {
            lock = true;
            if (loading) {
                removeCyclePrint();
            }
            appendText(message);
            lock = false;
        });

    }    

    private void appendText(String message) {

        int previousIndex = index - 1;
        if (previousIndex < 0) {
            previousIndex = cycle.length - 1;
        }
        //reset the index
        index = 0;
        logTextArea.append(message);
        logTextArea.append(System.lineSeparator());
        logTextArea.setCaretPosition(logTextArea.getText().length());
        logTextArea.validate();
        logTextArea.repaint();
    }

    private void removeCyclePrint() {
        Document doc = null;
        if (doc == null) {
            doc = logTextArea.getDocument();
        }
        try {
            doc.remove(doc.getLength() - (cycle[index].length()), cycle[index].length());
        } catch (BadLocationException ex) {
            //ignore
        }
    }

    @Override
    public void run() {
        while (true) {
            if (closed) {
                break;
            } else if (loading) {
                //check that nothing is writing to the logging area
                if (!lock && logTextArea != null) {
                    removeCyclePrint();
                    index++;
                    if (index > cycle.length - 1) {
                        index = 0;
                    }
                    String txt = logTextArea.getText();

                    logTextArea.setText(txt + cycle[index]);
                }

            }
            try {
                Thread.sleep(1000);
            } catch (InterruptedException ex) {
                //ignore
            }
        }
    }
}
