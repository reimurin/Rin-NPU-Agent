package com.geniex.demo.model

import android.app.AlertDialog
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.fragment.app.FragmentActivity
import com.geniex.demo.BuildConfig
import com.geniex.demo.R
import com.geniex.demo.databinding.ActivityModelLibraryBinding
import com.geniex.demo.utils.UiErrorLocalizer
import com.geniex.sdk.ModelManagerWrapper
import com.geniex.sdk.bean.HubSource
import com.geniex.sdk.bean.ModelPullInput
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class ModelLibraryActivity : FragmentActivity() {
    private lateinit var binding: ActivityModelLibraryBinding
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val hf = HuggingFaceSearchClient()
    private var searchResults: List<CatalogModel> = emptyList()
    private var installed: List<String> = emptyList()
    private var autoInstallHandled = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityModelLibraryBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.tvBuildMode.text = if (BuildConfig.TEST_MODEL_SEED) getString(R.string.test_build_badge) else ""
        bindRecommended()
        refreshActiveModel()
        refreshInstalled()
        binding.btnRefreshModels.setOnClickListener { refreshInstalled() }
        binding.btnSearchModels.setOnClickListener { binding.etModelSearch.text?.toString()?.trim()?.takeIf { it.isNotEmpty() }?.let(::search) }
        binding.btnQuickInstallPrimary.setOnClickListener { installRecommended(RecommendedModels.primary) }
        binding.btnQuickInstallVision.setOnClickListener { installRecommended(RecommendedModels.vision) }
        binding.listRecommended.setOnItemClickListener { _, _, pos, _ -> showModelDialog(RecommendedModels.all[pos]) }
        binding.listSearchResults.setOnItemClickListener { _, _, pos, _ -> if (pos < searchResults.size) showModelDialog(searchResults[pos]) }
        binding.listInstalled.setOnItemClickListener { _, _, pos, _ -> if (pos < installed.size) showInstalledDialog(installed[pos]) }
    }

    private fun bindRecommended() {
        val rows = RecommendedModels.all.map { m ->
            val badge = when (m.repoId) {
                RecommendedModels.primary.repoId -> getString(R.string.recommend_primary_npu)
                RecommendedModels.vision.repoId -> getString(R.string.recommend_vision_npu)
                else -> getString(R.string.recommend_fallback)
            }
            val runtime = if (m.runtime == "qairt") "NPU · SM8750" else (m.quant ?: "GGUF")
            "${m.displayName} · $badge\n${if (m.multimodal) getString(R.string.multimodal) else getString(R.string.text_only)} · $runtime"
        }
        binding.listRecommended.adapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, rows)
    }

    private fun search(query: String) {
        binding.tvDownloadStatus.text = "${getString(R.string.huggingface_search)}: $query"
        scope.launch {
            runCatching { hf.search(query) }.onSuccess { result ->
                searchResults = result
                runOnUiThread {
                    val rows = if (result.isEmpty()) listOf(getString(R.string.no_search_results)) else result.map { item ->
                        val kind = if (item.compatible) getString(R.string.compatible_model) else getString(R.string.source_only_model)
                        "${item.repoId}\n$kind · ↓${item.downloads}"
                    }
                    binding.listSearchResults.adapter = ArrayAdapter(this@ModelLibraryActivity, android.R.layout.simple_list_item_1, rows)
                }
            }.onFailure { e ->
                Log.e(TAG, "Hugging Face search failed", e)
                runOnUiThread { binding.tvDownloadStatus.text = getString(R.string.search_failed, UiErrorLocalizer.message(this@ModelLibraryActivity, e.message)) }
            }
        }
    }

    private fun showModelDialog(model: CatalogModel) {
        val message = buildString {
            append(model.repoId).append('\n')
            model.quant?.takeIf { it.isNotBlank() }?.let { append(getString(R.string.recommended_quant, it)).append('\n') }
            append(if (model.multimodal) getString(R.string.multimodal) else getString(R.string.text_only))
            append("\n").append(if (model.runtime == "qairt") getString(R.string.npu_runtime_sm8750) else getString(R.string.gguf_runtime_fallback))
            if (!model.compatible) append("\n\n").append(getString(R.string.gguf_required))
        }
        val dialog = AlertDialog.Builder(this).setTitle(model.displayName).setMessage(message).setNegativeButton(android.R.string.cancel, null)
        if (model.compatible) dialog.setPositiveButton(R.string.download_model) { _, _ -> installRecommended(model) }
        dialog.show()
    }

    private fun installRecommended(model: CatalogModel) {
        if (isInstalled(model.repoId)) {
            selectModel(model.repoId)
            Toast.makeText(this, R.string.model_already_downloaded, Toast.LENGTH_SHORT).show()
            updateQuickSetupState()
            return
        }
        download(model)
    }

    private fun download(model: CatalogModel) {
        binding.progressDownload.visibility = View.VISIBLE
        binding.progressDownload.progress = 0
        binding.tvDownloadStatus.text = getString(R.string.download_progress, model.repoId, 0)
        binding.btnQuickInstallPrimary.isEnabled = false
        binding.btnQuickInstallVision.isEnabled = false
        scope.launch {
            val hub = runCatching { HubSource.valueOf(model.hub) }.getOrDefault(HubSource.AUTO)
            val input = ModelPullInput(
                model_name = model.repoId,
                precision = model.quant,
                hub = hub,
                chipset = model.chipset,
            )
            ModelManagerWrapper.pullFlow(input).collect { event ->
                when (event) {
                    is ModelManagerWrapper.PullEvent.Progress -> {
                        val total = event.files.sumOf { if (it.total_bytes > 0) it.total_bytes else 0L }
                        val done = event.files.sumOf { it.downloaded_bytes }
                        val pct = if (total > 0) ((done * 100L) / total).toInt().coerceIn(0, 100) else 0
                        runOnUiThread {
                            binding.progressDownload.progress = pct
                            binding.tvDownloadStatus.text = getString(R.string.download_progress, model.displayName, pct)
                        }
                    }
                    is ModelManagerWrapper.PullEvent.Completed -> {
                        val paths = ModelManagerWrapper.getPaths(model.repoId)
                        selectModel(model.repoId)
                        runOnUiThread {
                            binding.progressDownload.visibility = View.GONE
                            binding.tvDownloadStatus.text = getString(R.string.download_complete_selected) + "\n" + getString(R.string.download_path, paths?.model_dir ?: "(unknown)")
                            Toast.makeText(this@ModelLibraryActivity, R.string.download_complete_selected, Toast.LENGTH_LONG).show()
                            refreshInstalled()
                        }
                    }
                    is ModelManagerWrapper.PullEvent.Error -> {
                        Log.e(TAG, "Model download failed: ${event.message}")
                        runOnUiThread {
                            binding.progressDownload.visibility = View.GONE
                            binding.tvDownloadStatus.text = getString(R.string.download_failed, UiErrorLocalizer.message(this@ModelLibraryActivity, event.message))
                            binding.btnQuickInstallPrimary.isEnabled = true
                            binding.btnQuickInstallVision.isEnabled = true
                        }
                    }
                }
            }
        }
    }

    private fun refreshInstalled() {
        scope.launch {
            installed = runCatching { ModelManagerWrapper.list() }.getOrDefault(emptyList())
            runOnUiThread {
                val rows = if (installed.isEmpty()) listOf(getString(R.string.no_installed_models)) else installed
                binding.listInstalled.adapter = ArrayAdapter(this@ModelLibraryActivity, android.R.layout.simple_list_item_1, rows)
                updateQuickSetupState()
                maybeHandleAutoInstall()
            }
        }
    }

    private fun updateQuickSetupState() {
        val primaryReady = isInstalled(RecommendedModels.primary.repoId)
        val visionReady = isInstalled(RecommendedModels.vision.repoId)
        binding.tvQuickSetupStatus.text = when {
            primaryReady && visionReady -> getString(R.string.quick_setup_both_ready)
            primaryReady -> getString(R.string.quick_setup_primary_ready)
            visionReady -> getString(R.string.quick_setup_vision_ready)
            else -> getString(R.string.quick_setup_not_installed)
        }
        binding.btnQuickInstallPrimary.text = getString(if (primaryReady) R.string.enable_primary_npu else R.string.install_primary_npu)
        binding.btnQuickInstallVision.text = getString(if (visionReady) R.string.enable_vision_npu else R.string.install_vision_npu)
        binding.btnQuickInstallPrimary.isEnabled = true
        binding.btnQuickInstallVision.isEnabled = true
    }

    private fun maybeHandleAutoInstall() {
        if (autoInstallHandled) return
        val repo = intent.getStringExtra(EXTRA_AUTO_INSTALL_REPO) ?: return
        autoInstallHandled = true
        val model = RecommendedModels.all.firstOrNull { sameModel(it.repoId, repo) } ?: return
        installRecommended(model)
    }

    private fun isInstalled(repo: String): Boolean = installed.any { sameModel(it, repo) }

    private fun sameModel(a: String, b: String): Boolean =
        a.equals(b, ignoreCase = true) || a.substringAfterLast('/').equals(b.substringAfterLast('/'), ignoreCase = true)

    private fun showInstalledDialog(modelName: String) {
        scope.launch {
            val paths = ModelManagerWrapper.getPaths(modelName)
            runOnUiThread {
                AlertDialog.Builder(this@ModelLibraryActivity)
                    .setTitle(modelName)
                    .setMessage(paths?.model_dir?.let { getString(R.string.download_path, it) } ?: modelName)
                    .setNegativeButton(android.R.string.cancel, null)
                    .setPositiveButton(R.string.enable_model) { _, _ ->
                        selectModel(modelName)
                        updateQuickSetupState()
                        Toast.makeText(this@ModelLibraryActivity, getString(R.string.model_enabled), Toast.LENGTH_SHORT).show()
                    }.show()
            }
        }
    }

    private fun selectModel(modelName: String) {
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(KEY_SELECTED_MODEL, modelName).apply()
        refreshActiveModel()
    }

    private fun refreshActiveModel() {
        val current = getSharedPreferences(PREFS, MODE_PRIVATE).getString(KEY_SELECTED_MODEL, null)
        binding.tvActiveModel.text = if (current.isNullOrBlank()) getString(R.string.active_model_none) else getString(R.string.active_model, current)
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "ModelLibrary"
        const val PREFS = "rin_model_library"
        const val KEY_SELECTED_MODEL = "selected_model"
        const val EXTRA_AUTO_INSTALL_REPO = "auto_install_repo"
    }
}
