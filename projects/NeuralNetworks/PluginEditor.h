#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class NeuralNetworksProcessorEditor : public juce::AudioProcessorEditor
{
public:
    NeuralNetworksProcessorEditor(NeuralNetworksProcessor&);
    ~NeuralNetworksProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    NeuralNetworksProcessor& audioProcessor;
    mrta::GenericParameterEditor genericParameterEditor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralNetworksProcessorEditor)
};
