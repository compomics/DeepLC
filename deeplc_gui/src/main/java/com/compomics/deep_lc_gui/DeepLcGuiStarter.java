package com.compomics.deep_lc_gui;

import com.compomics.deep_lc_gui.view.MainController;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.Painter;
import javax.swing.UIManager;
import org.apache.log4j.Logger;

/**
 * This class runs the pep_lc user interface (GUI).
 *
 * @author Niels Hulstaert
 */
public class DeepLcGuiStarter {

    /**
     * Logger instance.
     */
    private static final Logger LOGGER = Logger.getLogger(DeepLcGuiStarter.class);

    /**
     * The startup error message.
     */
    private static final String ERROR_MESSAGE = "An error occured during startup, please try again."
            + System.lineSeparator() + "If the problem persists, contact your administrator or post an issue on the google code page.";

    /**
     * The GUI main controller.
     */
    private final MainController mainController = new MainController();

    /**
     * No-arg constructor.
     */
    public DeepLcGuiStarter() {
    }

    /**
     * Main method.
     *
     * @param args the main method arguments
     */
    public static void main(final String[] args) {
        /*
         * Set the Nimbus look and feel
         */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /*
         * If Nimbus (introduced in Java SE 6) is not available, stay with the
         * default look and feel. For details see
         * http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html
         */
        try {
            for (UIManager.LookAndFeelInfo info : UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException | InstantiationException | IllegalAccessException | javax.swing.UnsupportedLookAndFeelException ex) {
            LOGGER.error(ex.getMessage(), ex);
        }
        //</editor-fold>

        //set background color for JOptionPane and JPanel instances
        UIManager.getLookAndFeelDefaults().put("OptionPane.background", Color.WHITE);
        UIManager.getLookAndFeelDefaults().put("Panel.background", Color.WHITE);
        UIManager.getLookAndFeelDefaults().put("FileChooser.background", Color.WHITE);
        //set background color for JFileChooser instances
        UIManager.getLookAndFeelDefaults().put("FileChooser[Enabled].backgroundPainter",
                (Painter<JFileChooser>) new Painter<JFileChooser>() {
                    @Override
                    public void paint(Graphics2D g, JFileChooser object, int width, int height) {
                        g.setColor(Color.WHITE);
                        g.draw(object.getBounds());
                    }
                });

        DeepLcGuiStarter deepLcGuiStarter = new DeepLcGuiStarter();
        deepLcGuiStarter.launch();
    }

    /**
     * Launch the GUI.
     */
    private void launch() {
        try {
            mainController.init();
            mainController.showView();
        } catch (Exception ex) {
            LOGGER.error(ex.getMessage(), ex);
            //add message to JTextArea
            JTextArea textArea = new JTextArea(ERROR_MESSAGE + System.lineSeparator() + System.lineSeparator() + ex.getMessage());
            //put JTextArea in JScrollPane
            JScrollPane scrollPane = new JScrollPane(textArea);
            scrollPane.setPreferredSize(new Dimension(600, 200));
            textArea.setEditable(false);
            textArea.setLineWrap(true);
            textArea.setWrapStyleWord(true);

            JOptionPane.showMessageDialog(null, scrollPane, "deep_lc GUI startup error", JOptionPane.ERROR_MESSAGE);
            System.exit(0);
        }
    }
}
