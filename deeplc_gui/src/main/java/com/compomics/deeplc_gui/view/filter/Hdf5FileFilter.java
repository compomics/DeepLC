package com.compomics.deeplc_gui.view.filter;

import java.io.File;
import javax.swing.filechooser.FileFilter;
import org.apache.commons.io.FilenameUtils;

/**
 * A file filter for HDF5 files.
 *
 * @author Niels Hulstaert
 */
public class Hdf5FileFilter extends FileFilter {

    public static final String HDF5_EXTENSION = "hdf5";
    public static final String H5_EXTENSION = "h5";
    private static final String DESCRIPTION = "*.hdf5";

    @Override
    public boolean accept(File file) {
        boolean accept = false;

        if (file.isFile()) {
            String extension = FilenameUtils.getExtension(file.getName());
            if (!extension.isEmpty() && (extension.equalsIgnoreCase(HDF5_EXTENSION) || extension.equalsIgnoreCase(H5_EXTENSION))) {
                accept = true;
            }
        } else {
            accept = true;
        }

        return accept;
    }

    @Override
    public String getDescription() {
        return DESCRIPTION;
    }

}
