/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.compomics.pep_lc_gui.view;

import java.io.*;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.text.Style;
import javax.swing.text.StyleConstants;
import javax.swing.text.StyledDocument;

/**
 * https://www.researchgate.net/post/print_the_output_of_a_processbuilder_in_a_java_TextArea_during_its_execution
 */
public class Console extends JTextArea implements Runnable {

    private JTextArea textArea;

    private Thread stdOutReader;

    private Thread stdErrReader;

    private boolean stopThreads;

    private final PipedInputStream stdOutPin = new PipedInputStream();

    private final PipedInputStream stdErrPin = new PipedInputStream();

    //Used to print error messages in red
    private StyledDocument doc;
    private Style style;

    /**
     * Initializes a new console
     */
    public Console() {
        // The area to which the output will be send to
        textArea = new JTextArea();
        textArea.setEditable(false);
        textArea.setBackground(Color.BLUE);
    }

    @Override
    public void run() {
        try {
            final ProcessBuilder processBuilder = new ProcessBuilder("ls", "-al");
            final Process process = processBuilder.start();
            final InputStreamReader inputStreamReader = new InputStreamReader(process.getInputStream());
            while (appendText(inputStreamReader)) {
                ;
            }
            process.waitFor();
            process.destroy();
        } catch (final Exception ex) {
            ex.printStackTrace();
        }
    }

    private boolean appendText(final InputStreamReader inputStreamReader) {
        try {
            final char[] buf = new char[256];
            final int read = inputStreamReader.read(buf);
            if (read < 1) {
                return false;
            }
            SwingUtilities.invokeLater(() -> {
                String jdjd = new String(buf);
                System.out.println("        " + jdjd);
                textArea.append(jdjd);
                textArea.append(System.lineSeparator());
                textArea.validate();
                textArea.repaint();
            });
            return true;
        } catch (final IOException e) {
            e.printStackTrace();
        }
        return false;
    }

    public void ping() {
        new Thread(this).start();
    }
}
