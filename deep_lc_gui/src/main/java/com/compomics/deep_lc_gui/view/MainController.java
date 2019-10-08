package com.compomics.deep_lc_gui.view;

import com.compomics.deep_lc_gui.config.ConfigHolder;
import java.awt.CardLayout;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Writer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.stream.Collectors;
import javax.swing.DefaultListModel;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingWorker;
import org.apache.commons.lang.SystemUtils;
import org.apache.log4j.Logger;

/**
 * The GUI main controller.
 *
 * @author Niels Hulstaert
 */
public class MainController {

    /**
     * Logger instance.
     */
    private static final Logger LOGGER = Logger.getLogger(MainController.class);

    //card layout panel names
    private static final String FIRST_PANEL = "firstPanel";
    private static final String LAST_PANEL = "lastPanel";

    /**
     * Model fields.
     */
    /**
     * The prediction peptides file.
     */
    private File predictionPeptidesFile;
    /**
     * The calibration peptides file.
     */
    private File calibrationPeptidesFile;
    /**
     * The model file list model.
     */
    private final DefaultListModel<File> modelFileListModel = new DefaultListModel<>();
    /**
     * The pep lc output file.
     */
    private File outPutFile;
    /**
     * The swing worker.
     */
    DeepLcSwingWorker deepLcSwingWorker;
    /**
     * The views of this controller.
     */
    private final MainFrame mainFrame = new MainFrame();
    private LogTextAreaAppender logTextAreaAppender;

    /**
     * Initialize the controller.
     */
    public void init() {
        //select files only and file filters
        mainFrame.getPredictionPeptidesChooser().setFileSelectionMode(JFileChooser.FILES_ONLY);
        //mainFrame.getPredictionPeptidesChooser().setFileFilter(new TabSeparatedFileFilter());
        mainFrame.getCalibrationPeptidesChooser().setFileSelectionMode(JFileChooser.FILES_ONLY);
        //mainFrame.getCalibrationPeptidesChooser().setFileFilter(new TabSeparatedFileFilter());
        mainFrame.getModelFileChooser().setFileSelectionMode(JFileChooser.FILES_ONLY);
        //mainFrame.getModelFileChooser().setFileFilter(new Hdf5FileFilter());
        mainFrame.getModelFileChooser().setMultiSelectionEnabled(true);
        mainFrame.getOutputFileChooser().setFileSelectionMode(JFileChooser.FILES_ONLY);

        mainFrame.setTitle("DeepLC " + ConfigHolder.getInstance().getString("deep_lc_gui.version", "N/A"));

        //OutLogger.tieSystemOutAndErrToLog();

        //get the gui appender for setting the log text area
        logTextAreaAppender = (LogTextAreaAppender) Logger.getRootLogger().getAppender("gui");
        logTextAreaAppender.setLogTextArea(mainFrame.getLogTextArea());

        mainFrame.getLogTextArea().setText("..." + System.lineSeparator());

        //show info
        updateInfo("Click on \"start\" to run the ???.");

        //add action listeners
        mainFrame.getPredictionPeptidesButton().addActionListener(e -> {
            int returnVal = mainFrame.getPredictionPeptidesChooser().showOpenDialog(mainFrame);
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                predictionPeptidesFile = mainFrame.getPredictionPeptidesChooser().getSelectedFile();
                mainFrame.getPredictionPeptidesTextField().setText(predictionPeptidesFile.getAbsolutePath());
            }
        });

        mainFrame.getCalibrationPeptidesButton().addActionListener(e -> {
            int returnVal = mainFrame.getCalibrationPeptidesChooser().showOpenDialog(mainFrame);
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                calibrationPeptidesFile = mainFrame.getCalibrationPeptidesChooser().getSelectedFile();
                mainFrame.getCalibrationPeptidesTextField().setText(calibrationPeptidesFile.getAbsolutePath());
            }
        });

        mainFrame.getModelList().setModel(modelFileListModel);
        mainFrame.getModelButton().addActionListener(e -> {
            int returnVal = mainFrame.getModelFileChooser().showOpenDialog(mainFrame);
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                File[] modelFiles = mainFrame.getModelFileChooser().getSelectedFiles();
                for (File modelFile : modelFiles) {
                    modelFileListModel.addElement(modelFile);
                }
            }
        });

        mainFrame.getRemoveModelButton().addActionListener(e -> {
            if (mainFrame.getModelList().getSelectedIndex() != -1) {
                mainFrame.getModelList().getSelectedValuesList().forEach((modelFileToRemove) -> {
                    DefaultListModel model = (DefaultListModel) mainFrame.getModelList().getModel();
                    model.removeElement(modelFileToRemove);
                });
            } else {
                List<String> messages = new ArrayList<>();
                messages.add("Please select one or more model files to delete");
                showMessageDialog("Model file removal", messages, JOptionPane.INFORMATION_MESSAGE);
            }
        });

        mainFrame.getOutputFileButton().addActionListener(e -> {
            int returnVal = mainFrame.getOutputFileChooser().showOpenDialog(mainFrame);
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                outPutFile = mainFrame.getOutputFileChooser().getSelectedFile();
                mainFrame.getOutputFileTextField().setText(outPutFile.getAbsolutePath());
            }
        });

        mainFrame.getClearButton().addActionListener(e -> {
            mainFrame.getLogTextArea().setText("");
            logTextAreaAppender.setLoading(true);
        });

        mainFrame.getStartButton().addActionListener(e -> {
            //List<String> firstPanelValidationMessages = validateFirstPanel();
            List<String> firstPanelValidationMessages = new ArrayList<>();
            if (firstPanelValidationMessages.isEmpty()) {
                getCardLayout().show(mainFrame.getTopPanel(), LAST_PANEL);
                onCardSwitch();

                //run the DeepLC worker
                deepLcSwingWorker = new DeepLcSwingWorker();
                deepLcSwingWorker.execute();
            } else {
                showMessageDialog("Validation failure", firstPanelValidationMessages, JOptionPane.WARNING_MESSAGE);
            }
        });

        mainFrame.getCancelButton().addActionListener(e -> {
            if (deepLcSwingWorker != null) {
                deepLcSwingWorker.cancel(true);
                logTextAreaAppender.close();
                deepLcSwingWorker = null;
            }
            mainFrame.dispose();
            System.exit(0);
        });

        //load the parameters from the properties file
        loadParameterValues();

        //call onCardSwitch
        onCardSwitch();
    }

    /**
     * Show the view of this controller.
     */
    public void showView() {
        mainFrame.setLocationRelativeTo(null);
        mainFrame.setVisible(true);
    }

    /**
     * Load the parameter values from the properties file and set them in the
     * matching fields.
     */
    private void loadParameterValues() {
        mainFrame.getNumberOfThreadsTextField().setText(ConfigHolder.getInstance().getString("number_of_threads"));
        mainFrame.getSplitCalibrationTextField().setText(ConfigHolder.getInstance().getString("split_calibration"));
        mainFrame.getDictionaryDividerTextField().setText(ConfigHolder.getInstance().getString("dictionary_divider"));
        mainFrame.getBatchNumberTextField().setText(ConfigHolder.getInstance().getString("batch_number"));
    }

    /**
     * Show a message dialog with a text area if the messages list contains more
     * than one message or if the message is an error message.
     *
     * @param title the dialog title
     * @param messages the dialog messages list
     * @param messageType the dialog message type
     */
    private void showMessageDialog(final String title, final List<String> messages, final int messageType) {
        if (messages.size() > 1 || messageType == JOptionPane.ERROR_MESSAGE) {
            String message = messages.stream().collect(Collectors.joining(System.lineSeparator()));

            //add message to JTextArea
            JTextArea textArea = new JTextArea(message);
            //put JTextArea in JScrollPane
            JScrollPane scrollPane = new JScrollPane(textArea);
            scrollPane.setPreferredSize(new Dimension(600, 200));
            scrollPane.getViewport().setOpaque(false);
            textArea.setEditable(false);
            textArea.setLineWrap(true);
            textArea.setWrapStyleWord(true);

            JOptionPane.showMessageDialog(mainFrame.getContentPane(), scrollPane, title, messageType);
        } else {
            JOptionPane.showMessageDialog(mainFrame.getContentPane(), messages.get(0), title, messageType);
        }
    }

    /**
     * Show the correct info and disable/enable the right buttons when switching
     * between cards.
     */
    private void onCardSwitch() {
        String currentCardName = getVisibleChildComponent(mainFrame.getTopPanel());
        switch (currentCardName) {
            case FIRST_PANEL:
                //show info
                updateInfo("Click on \"start\" to run Deep LC");
                break;
            case LAST_PANEL:
                mainFrame.getStartButton().setEnabled(false);
                //show info
                updateInfo("");
                break;
            default:
                break;
        }
    }

    /**
     * Update the info label.
     *
     * @param message the info message
     */
    private void updateInfo(String message) {
        mainFrame.getInfoLabel().setText(message);
    }

    /**
     * Validate the user input in the first panel.
     *
     * @return the list of validation messages.
     */
    private List<String> validateFirstPanel() {
        List<String> validationMessages = new ArrayList<>();

        if (predictionPeptidesFile == null || !predictionPeptidesFile.exists()) {
            validationMessages.add("Please choose a valid predictions peptides file.");
        }
        if (calibrationPeptidesFile == null || !calibrationPeptidesFile.exists()) {
            validationMessages.add("Please choose a valid calibration peptides file.");
        }
        if (modelFileListModel.isEmpty()) {
            validationMessages.add("Please choose at least one model file.");
        } else {
            Enumeration<File> elements = modelFileListModel.elements();
            while (elements.hasMoreElements()) {
                File modelFile = elements.nextElement();
                if (!modelFile.exists()) {
                    validationMessages.add("Please choose a valid model file: " + modelFile.getAbsolutePath());
                }
            }
        }
        try {
            Integer dictionaryDivider = Integer.valueOf(mainFrame.getDictionaryDividerTextField().getText());
            if (dictionaryDivider < 0) {
                validationMessages.add("Please provide a positive dictionary divider value.");
            }
        } catch (NumberFormatException nfe) {
            validationMessages.add("Please provide a numeric dictionary divider value.");
        }
        try {
            Integer numberOfThreads = Integer.valueOf(mainFrame.getNumberOfThreadsTextField().getText());
            if (numberOfThreads < 0) {
                validationMessages.add("Please provide a positive number of threads.");
            }
        } catch (NumberFormatException nfe) {
            validationMessages.add("Please provide a numeric number of threads value.");
        }
        try {
            Integer splitCalibration = Integer.valueOf(mainFrame.getSplitCalibrationTextField().getText());
            if (splitCalibration < 0) {
                validationMessages.add("Please provide a positive split calibration value.");
            }
        } catch (NumberFormatException nfe) {
            validationMessages.add("Please provide a numeric split calibration value.");
        }
        try {
            Integer splitCalibration = Integer.valueOf(mainFrame.getBatchNumberTextField().getText());
            if (splitCalibration < 0) {
                validationMessages.add("Please provide a bacth number value.");
            }
        } catch (NumberFormatException nfe) {
            validationMessages.add("Please provide a numeric batch number value.");
        }

        return validationMessages;
    }

    /**
     * Get the name of the visible child component. Returns null if no
     * components are visible.
     *
     * @param parentContainer the parent container
     * @return the visible component name
     */
    private String getVisibleChildComponent(final Container parentContainer) {
        String visibleComponentName = null;

        for (Component component : parentContainer.getComponents()) {
            if (component.isVisible()) {
                visibleComponentName = component.getName();
                break;
            }
        }

        return visibleComponentName;
    }

    /**
     * Get the card layout.
     *
     * @return the CardLayout
     */
    private CardLayout getCardLayout() {
        return (CardLayout) mainFrame.getTopPanel().getLayout();
    }

    /**
     * DeepLC Swing worker for running moFF.
     */
    private class DeepLcSwingWorker extends SwingWorker<Void, Void> {

        /**
         * No-arg constructor.
         */
        public DeepLcSwingWorker() {

        }

        @Override
        protected Void doInBackground() throws Exception {
            LOGGER.info("Starting to run Deep LC...");

            // start the waiting animation
            logTextAreaAppender.setLoading(true);

            File tempScript;
            ProcessBuilder pb = null;
            if (SystemUtils.IS_OS_LINUX) {
                tempScript = createLinuxTempScript();
                pb = new ProcessBuilder("bash", tempScript.toString());
            } else if (SystemUtils.IS_OS_WINDOWS) {
                tempScript = createWindowsTempScript();
                pb = new ProcessBuilder(tempScript.getAbsolutePath());
            } else {
                throw new UnsupportedOperationException();
            }

            // Start the process.
            Process process = pb.start();

            InputStream out = process.getInputStream();
            byte[] buffer = new byte[4000];
            while (isAlive(process)) {
                int no = out.available();
                if (no > 0) {
                    int n = out.read(buffer, 0, Math.min(no, buffer.length));
                    //String output = 
                    LOGGER.info(new String(buffer, 0, n));
                }

                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                }
            }

            // stop the waiting animation
            logTextAreaAppender.setLoading(false);
            return null;
        }

        @Override
        protected void done() {
            try {
                get();
                LOGGER.info("Finished DeepLC run.");
                List<String> messages = new ArrayList<>();
                messages.add("The DeepLC run has finished successfully.");
                showMessageDialog("DeepLC run completed", messages, JOptionPane.INFORMATION_MESSAGE);
            } catch (CancellationException ex) {
                LOGGER.info("The DeepLC run was cancelled.");
            } catch (Exception ex) {
                ex.printStackTrace();
                List<String> messages = new ArrayList<>();
                messages.add(ex.getMessage());
                showMessageDialog("Unexpected error", messages, JOptionPane.ERROR_MESSAGE);
                logTextAreaAppender.setLoading(false);
            } finally {

            }
        }

        private File createLinuxTempScript() throws IOException {
            File tempScript = File.createTempFile("script", null);

            try (Writer streamWriter = new OutputStreamWriter(new FileOutputStream(
                    tempScript));
                    PrintWriter printWriter = new PrintWriter(streamWriter)) {
                printWriter.println("#!/bin/bash");

                String deep_lc_location = ConfigHolder.getInstance().getString("deep_lc_location_linux");
                StringBuilder command = new StringBuilder();
                command.append(ConfigHolder.getInstance().getString("conda_env_location_linux")).append("/bin/python").append(" ");
                command.append(deep_lc_location).append("/run.py");
                command.append(" --file_pred ").append(deep_lc_location).append("/datasets/test_pred.csv");
                //command.append(" --file_pred ").append(predictionPeptidesFile.getAbsolutePath());
                command.append(" --file_cal ").append(deep_lc_location).append("/datasets/test_train.csv");
                //command.append(" --file_cal ").append(calibrationPeptidesFile.getAbsolutePath());
                command.append(" --file_pred_out ").append(deep_lc_location).append("/datasets/preds_out.csv");
                //command.append(" --file_pred_out ").append(outPutFile.getAbsolutePath());
                command.append(" --file_model ").append(deep_lc_location).append("/mods/full_dia.hdf5");
                //command.append(" --file_model ");
                //Enumeration<File> elements = modelFileListModel.elements();
                //while (elements.hasMoreElements()) {
                //    File modelFile = elements.nextElement();
                //    command.append(modelFile.getAbsolutePath());
                //}
                command.append(" --n_threads ").append(mainFrame.getNumberOfThreadsTextField().getText());
                command.append(" --split_cal ").append(mainFrame.getSplitCalibrationTextField().getText());
                command.append(" --dict_divider ").append(mainFrame.getDictionaryDividerTextField().getText());
                command.append(" --dict_divider ").append(mainFrame.getBatchNumberTextField().getText());

                printWriter.println(command);
            }

            return tempScript;
        }

        private File createWindowsTempScript() throws IOException {
            File tempScript = File.createTempFile("script", ".bat");

            try (Writer streamWriter = new OutputStreamWriter(new FileOutputStream(
                    tempScript));
                    PrintWriter printWriter = new PrintWriter(streamWriter)) {
                StringBuilder command = new StringBuilder();
                command.append("call ");
                String condaEnvLocation = ConfigHolder.getInstance().getString("conda_env_location_windows");      
                if (!new File(condaEnvLocation).isAbsolute()) {
                    Path currentRelativePath = Paths.get("");
                    String currentAbsolutePath = currentRelativePath.toAbsolutePath().toString();
                    command.append(currentAbsolutePath).append("/");
                }
                command.append(condaEnvLocation).append("/Scripts/activate.bat DL & ^python ");
                command.append("../run.py");
                
                String deepLcLocation = ConfigHolder.getInstance().getString("deep_lc_location_windows");
                command.append(" --file_pred ").append(deepLcLocation).append("/datasets/test_pred.csv");
                //command.append(" --file_pred ").append(predictionPeptidesFile.getAbsolutePath());
                command.append(" --file_cal ").append(deepLcLocation).append("/datasets/test_train.csv");
                //command.append(" --file_cal ").append(calibrationPeptidesFile.getAbsolutePath());                
                command.append(" --file_pred_out ").append(deepLcLocation).append("/datasets/preds_out.csv");
                //command.append(" --file_pred_out ").append(outPutFile.getAbsolutePath());                
                command.append(" --file_model ").append(deepLcLocation).append("/mods/full_dia.hdf5");
                //command.append(" --file_model ");
                //Enumeration<File> elements = modelFileListModel.elements();
                //while(elements.hasMoreElements()){
                //    File modelFile = elements.nextElement();
                //    command.append(modelFile.getAbsolutePath());
                //}
                command.append(" --n_threads ").append(mainFrame.getNumberOfThreadsTextField().getText());
                command.append(" --split_cal ").append(mainFrame.getSplitCalibrationTextField().getText());
                command.append(" --dict_divider ").append(mainFrame.getDictionaryDividerTextField().getText());
                command.append(" --dict_divider ").append(mainFrame.getBatchNumberTextField().getText());

                printWriter.print(command);
            }

            return tempScript;
        }

        private boolean isAlive(Process p) {
            try {
                p.exitValue();
                return false;
            } catch (IllegalThreadStateException e) {
                return true;
            }
        }
    }
}
