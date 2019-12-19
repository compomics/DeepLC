package com.compomics.deeplc_gui.config;

import java.io.File;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;

/**
 *
 * @author Niels Hulstaert
 */
public class ResourceUtils {

    private static final String RESOURCES_FOLDER = "resources";

    /**
     * Gets a resource by its relative path. If the resource is not found on the
     * file system, the classpath is searched. If nothing is found, null is
     * returned.
     *
     * @param fileName the name of the resource
     * @return the found resource
     */
    public static Resource getResourceByRelativePath(String fileName) {
        Resource resource = new FileSystemResource(fileName);
        if (!resource.exists()) {
            //try to find it on the classpath
            resource = new ClassPathResource(fileName);
            if (!resource.exists()) {
                // making sure to run on Netbeans..
                resource = new FileSystemResource("src" + File.separator + "main" + File.separator + RESOURCES_FOLDER + File.separator + fileName);
                if (!resource.exists()) {
                    resource = null;
                }
            }
        }
        return resource;
    }

    /**
     * Checks if a resource with the given relative path exists on the file
     * system.
     *
     * @param relativePath the relative path of the resource
     * @return the is existing boolean
     */
    public static boolean isExistingFile(String relativePath) {
        boolean isExistingResource = Boolean.FALSE;
        Resource resource = new FileSystemResource(relativePath);
        if (resource.exists()) {
            isExistingResource = true;
        }

        return isExistingResource;
    }
}
