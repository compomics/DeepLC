package com.compomics.deep_lc_gui.config;

import java.io.IOException;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.log4j.Logger;
import org.springframework.core.io.Resource;

/**
 * This class is a PropertiesConfiguration singleton.
 */
public class ConfigHolder extends PropertiesConfiguration {

    /**
     * Logger instance.
     */
    private static final Logger LOGGER = Logger.getLogger(ConfigHolder.class);
    /**
     * ConfigHolder singleton instance.
     */
    private static ConfigHolder ourInstance;

    static {
        try {
            Resource propertiesResource = ResourceUtils.getResourceByRelativePath("deep_lc_gui.properties");
            ourInstance = new ConfigHolder(propertiesResource);
        } catch (IOException | ConfigurationException e) {
            LOGGER.error(e.getMessage(), e);
        }
    }

    /**
     * Private constructor to prevent instantiation.
     *
     * @param propertiesResource the properties resource
     * @throws ConfigurationException in case of a configuration initialization
     * problem
     * @throws IOException in case of an I/O related problem
     */
    private ConfigHolder(Resource propertiesResource) throws ConfigurationException, IOException {
        super(propertiesResource.getURL());
    }

    /**
     * Gets the PropertiesConfiguration instance
     *
     * @return the PropertiesConfigurationHolder instance
     */
    public static ConfigHolder getInstance() {
        return ourInstance;
    }

}
